"""Interactive backdoor explorer: pick a dataset and a class, draw a random sample, and see it before and after the backdoor
trigger is added. The "after" image is produced by the SAME code the experiments use (data.prep -> data.make_trigger), not a
re-implementation, so what you see is exactly what the clients train on.

    from fedup_repro import viz
    viz.backdoor_explorer()                                  # widgets (Kaggle / Colab / Jupyter)
    viz.show_backdoor("CIFAR-10", class_id=3, seed=1)        # one static figure
"""
import numpy as np
import torch

from . import downloaders
from .config import BASE, make_config
from .data import make_trigger, norm_stats, prep

CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CIFAR100 = ("apple aquarium_fish baby bear beaver bed bee beetle bicycle bottle bowl boy bridge bus butterfly camel can castle "
            "caterpillar cattle chair chimpanzee clock cloud cockroach couch crab crocodile cup dinosaur dolphin elephant flatfish "
            "forest fox girl hamster house kangaroo keyboard lamp lawn_mower leopard lion lizard lobster man maple_tree motorcycle "
            "mountain mouse mushroom oak_tree orange orchid otter palm_tree pear pickup_truck pine_tree plain plate poppy porcupine "
            "possum rabbit raccoon ray road rocket rose sea seal shark shrew skunk skyscraper snail snake spider squirrel streetcar "
            "sunflower sweet_pepper table tank telephone television tiger tractor train trout tulip turtle wardrobe whale willow_tree "
            "wolf woman worm").split()
CLASS_NAMES = {"MNIST": [str(i) for i in range(10)], "CIFAR-10": CIFAR10, "CIFAR-100": CIFAR100,
               "AG News (text)": ["World", "Sports", "Business", "Sci/Tech"]}
DATASETS = {"MNIST": ("MNIST-LeNet5", "mnist"), "CIFAR-10": ("CIFAR10-ResNet18", "cifar10"),
            "CIFAR-100": ("CIFAR100-ResNet34", "cifar100"), "AG News (text)": ("AGNews-TinyBERT", "agnews")}
# label -> (scenario, trigger_mode)
IMAGE_TRIGGERS = {
    "FedUP sample trigger: 1 pixel (released code: 255 written in normalised space)": ("sample", "repo"),
    "FedUP sample trigger: 1 pixel (real white, 'paper' wording)": ("sample", "white"),
    "FuGuard patch: k x k white square, bottom-right": ("client_backdoor", "white"),
}
TEXT_TRIGGERS = {"FedUP text trigger: token 1996 ('the') written at the last position": ("sample", "repo")}
_POOLS = {}


def _pool(dataset, arrays=None):
    """-> (x, y, tokenizer or None) for the training split of a dataset (loaded once, cached in memory)."""
    if arrays and dataset in arrays:
        x, y = arrays[dataset][:2]
        return x, y, (arrays[dataset][2] if len(arrays[dataset]) > 2 else None)
    if dataset not in _POOLS:
        bench, key = DATASETS[dataset]
        if key == "agnews":
            tx, ty, _, _ = downloaders.load_agnews_arrays(BASE["text_model"], BASE["max_len"])
            try:
                tok = downloaders.load_tokenizer(BASE["text_model"])
            except Exception as e:  # noqa: BLE001
                print(f"[viz] tokenizer unavailable ({e!r}); showing token ids")
                tok = None
            _POOLS[dataset] = (tx, ty, tok)
        else:
            tx, ty, _, _ = downloaders.load_image_arrays(key)
            _POOLS[dataset] = (tx, ty, None)
    return _POOLS[dataset]


def backdoor_pair(img, benchmark, scenario, trigger_mode="repo", patch=4):
    """img: uint8 [C,H,W]. -> before, after (float [H,W,C] in [0,1]), target label, info dict. Uses the training code path."""
    cfg = make_config(benchmark, scenario, trigger_mode=trigger_mode, bd_patch=int(patch))
    x = torch.from_numpy(np.ascontiguousarray(img))[None]
    clean = prep(x, cfg, torch.device("cpu"))
    trig = make_trigger(cfg, scenario)(clean)
    mean, std = norm_stats(cfg)
    m, s = torch.tensor(mean).view(1, -1, 1, 1), torch.tensor(std).view(1, -1, 1, 1)

    def to_img(t):
        return (t * s + m).clamp(0, 1)[0].permute(1, 2, 0).numpy()

    before, after = to_img(clean), to_img(trig)
    target = int(cfg["sample_target"] if scenario == "sample" else cfg["bd_target"])
    changed = np.abs(after - before).max(-1) > 1e-6
    info = dict(changed_pixels=int(changed.sum()), image_size=tuple(before.shape[:2]), target=target,
                written_normalised_value=(255.0 if (scenario == "sample" and trigger_mode == "repo") else None))
    if info["written_normalised_value"] is not None:
        info["pixel_value_after_denormalising"] = [round(float(255.0 * sd + mu), 1) for mu, sd in zip(mean, std)]
    return before, after, target, info


def render_image(dataset, x, label, trigger_label, patch, arrays=None):
    """-> matplotlib Figure, info for one image."""
    import matplotlib.pyplot as plt
    bench, _ = DATASETS[dataset]
    scenario, mode = IMAGE_TRIGGERS[trigger_label]
    before, after, target, info = backdoor_pair(x, bench, scenario, mode, patch)
    names = CLASS_NAMES[dataset]
    diff = np.abs(after - before).max(-1)
    fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.9))
    gray = before.shape[-1] == 1
    z = 8                                                   # zoom on the bottom-right corner, where every trigger is written
    for a, im, title in ((ax[0], before, f"BEFORE\nlabel {label}: {names[label]}"),
                         (ax[1], after, f"AFTER (backdoor added)\nrelabelled -> {target}: {names[target]}"),
                         (ax[2], after[-z:, -z:], f"zoom: bottom-right {z}x{z} of AFTER")):
        a.imshow(im[..., 0] if gray else im, cmap="gray" if gray else None, vmin=0, vmax=1, interpolation="nearest")
        a.set_title(title, fontsize=9)
        a.axis("off")
    ax[3].imshow(diff, cmap="hot", vmin=0, vmax=max(diff.max(), 1e-6), interpolation="nearest")
    ax[3].set_title(f"difference map\nchanged pixels: {info['changed_pixels']}", fontsize=9)
    ax[3].axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, info


def render_text(x, label, tok, trigger_label, max_show=40, tail=10):
    """-> HTML string: the text before and after the trigger token is written at the last position of the padded sequence."""
    bench, _ = DATASETS["AG News (text)"]
    cfg = make_config(bench, "sample")
    ids, mask = x[0].copy(), x[1].copy()
    after = make_trigger(cfg, "sample")(torch.from_numpy(x.copy())[None])[0].numpy()
    n = int(mask.sum())

    def toks(a):
        return tok.convert_ids_to_tokens([int(i) for i in a]) if tok is not None else [str(int(i)) for i in a]

    head = toks(ids[:min(n, max_show)])
    real_tail = toks(ids[max(0, n - tail):n]) if n > max_show else []
    L = len(ids)
    trig_tok = toks([after[0][-1]])[0]
    pad_tok = toks([ids[-1]])[0] if mask[-1] == 0 else "(last real token)"
    names = CLASS_NAMES["AG News (text)"]
    html = (f"<div style='font-family:monospace;font-size:12px;line-height:1.5'>"
            f"<b>BEFORE</b> &nbsp;label {label}: {names[label]} &nbsp;|&nbsp; {n} real tokens, padded to {L}<br>"
            f"<span style='background:#eef'>{' '.join(head)}</span>"
            f"{' ... ' + ' '.join(real_tail) if real_tail else ''} "
            f"<span style='color:#999'>{pad_tok} x {L - n}</span><br><br>"
            f"<b>AFTER</b> &nbsp;label {label} -> {int(cfg['sample_target'])}: {names[int(cfg['sample_target'])]} (relabelled to the target class)<br>"
            f"<span style='background:#eef'>{' '.join(head)}</span>"
            f"{' ... ' + ' '.join(real_tail) if real_tail else ''} "
            f"<span style='color:#999'>{pad_tok} x {L - n - (1 if mask[-1] == 0 else 0)}</span> "
            f"<span style='background:#fbb;font-weight:bold'>{trig_tok}</span> "
            f"<span style='color:#777'>(position {L - 1}: id {int(after[0][-1])}, attention mask set to 1)</span></div>")
    return html


def show_backdoor(dataset="CIFAR-10", class_id=None, index=None, seed=0, trigger=None, patch=4, arrays=None):
    """One static before / after figure (or HTML for text). class_id=None picks any class; index picks a specific sample."""
    from IPython.display import HTML, display
    import matplotlib.pyplot as plt
    x, y, tok = _pool(dataset, arrays)
    rng = np.random.RandomState(seed)
    pool_idx = np.arange(len(y)) if class_id is None else np.where(np.asarray(y) == class_id)[0]
    i = int(index) if index is not None else int(rng.choice(pool_idx))
    label = int(y[i])
    if dataset.startswith("AG"):
        trigger = trigger or next(iter(TEXT_TRIGGERS))
        display(HTML(render_text(np.asarray(x[i]), label, tok, trigger)))
        return i
    trigger = trigger or next(iter(IMAGE_TRIGGERS))
    fig, info = render_image(dataset, np.asarray(x[i]), label, trigger, patch)
    print(f"{dataset} | sample #{i} | trigger: {trigger}")
    if info.get("written_normalised_value") is not None:
        print(f"  writes {info['written_normalised_value']:.0f} into the NORMALISED tensor = pixel value {info['pixel_value_after_denormalising']} "
              f"(shown saturated at 1.0)")
    plt.show()
    return i


def backdoor_explorer(arrays=None, seed=0):
    """Widgets: dataset, class, trigger, patch size and a 'random image' button. `arrays` (tests): {dataset: (x, y[, tokenizer])}."""
    import ipywidgets as W
    import matplotlib.pyplot as plt
    from IPython.display import HTML, clear_output, display
    rng = np.random.RandomState(seed)
    st = {"idx": None, "busy": False}
    dd_data = W.Dropdown(options=list(DATASETS), description="dataset", layout=W.Layout(width="260px"))
    dd_class = W.Dropdown(description="class", layout=W.Layout(width="320px"))
    dd_trig = W.Dropdown(description="trigger", layout=W.Layout(width="720px"))
    sl_patch = W.IntSlider(value=4, min=1, max=10, description="patch k", layout=W.Layout(width="260px"))
    btn = W.Button(description="random image", icon="random", button_style="primary")
    out = W.Output()

    def set_options():
        st["busy"] = True                                  # changing the options fires the observers: ignore them meanwhile
        d = dd_data.value
        dd_class.options = [("(any class)", -1)] + [(f"{i}: {n}", i) for i, n in enumerate(CLASS_NAMES[d])]
        dd_trig.options = list(TEXT_TRIGGERS if d.startswith("AG") else IMAGE_TRIGGERS)
        dd_class.index, dd_trig.index = 0, 0
        sl_patch.disabled = d.startswith("AG")
        st["busy"] = False

    def pick():
        x, y, _ = _pool(dd_data.value, arrays)
        y = np.asarray(y)
        cand = np.arange(len(y)) if dd_class.value == -1 else np.where(y == dd_class.value)[0]
        st["idx"] = int(rng.choice(cand))

    def draw():
        with out:
            clear_output(wait=True)
            d = dd_data.value
            try:
                x, y, tok = _pool(d, arrays)
            except Exception as e:  # noqa: BLE001
                print(f"could not load {d}: {e!r}")
                return
            if st["idx"] is None:
                pick()
            i = st["idx"]
            label = int(y[i])
            if d.startswith("AG"):
                display(HTML(f"<i>sample #{i}</i>" + render_text(np.asarray(x[i]), label, tok, dd_trig.value)))
                return
            fig, info = render_image(d, np.asarray(x[i]), label, dd_trig.value, sl_patch.value)
            print(f"{d} | sample #{i} of {len(y)} | trigger: {dd_trig.value}")
            if info.get("written_normalised_value") is not None:
                print(f"writes {info['written_normalised_value']:.0f} into the NORMALISED tensor = pixel value "
                      f"{info['pixel_value_after_denormalising']} (displayed saturated at 1.0)")
            plt.show()

    def on_dataset(_=None):
        if st["busy"]:
            return
        with out:
            clear_output(wait=True)
            print(f"loading {dd_data.value} (first time may download or tokenise) ...")
        set_options()
        pick()
        draw()

    def on_class(_=None):
        if st["busy"]:
            return
        pick()
        draw()

    def on_other(_=None):
        if st["busy"]:
            return
        draw()

    def on_button(_=None):
        pick()
        draw()

    set_options()
    dd_data.observe(on_dataset, names="value")
    dd_class.observe(on_class, names="value")
    dd_trig.observe(on_other, names="value")
    sl_patch.observe(on_other, names="value")
    btn.on_click(on_button)
    box = W.VBox([W.HBox([dd_data, dd_class]), dd_trig, W.HBox([sl_patch, btn]), out])
    display(box)
    on_dataset()
    return box
