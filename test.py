import argparse
import torch
import numpy as np
import sys

def summarize(obj, max_items=8, max_chars=200, indent=0):
    pad = " " * indent
    t = type(obj)
    if isinstance(obj, torch.Tensor):
        return f"{pad}Tensor shape={tuple(obj.shape)} dtype={obj.dtype} requires_grad={obj.requires_grad}"
    if isinstance(obj, np.ndarray):
        return f"{pad}ndarray shape={obj.shape} dtype={obj.dtype}"
    if isinstance(obj, dict):
        s = f"{pad}dict keys={len(obj)}"
        if len(obj) <= max_items:
            for k,v in obj.items():
                s += "\n" + pad + f"  [{repr(k)}] -> " + summarize(v, max_items, max_chars, indent+4).lstrip()
        else:
            s += f" (showing first {max_items} keys)\n"
            for i,k in enumerate(list(obj.keys())[:max_items]):
                s += "\n" + pad + f"  [{repr(k)}] -> " + summarize(obj[k], max_items, max_chars, indent+4).lstrip()
        return s
    if isinstance(obj, (list, tuple)):
        names = "list" if isinstance(obj, list) else "tuple"
        s = f"{pad}{names} len={len(obj)}"
        for i,el in enumerate(obj[:max_items]):
            s += "\n" + pad + f"  [{i}] -> " + summarize(el, max_items, max_chars, indent+4).lstrip()
        if len(obj) > max_items:
            s += f"\n{pad}  ... ({len(obj)-max_items} more)"
        return s
    # fallback: small repr
    r = repr(obj)
    if len(r) > max_chars:
        r = r[:max_chars] + "..."
    return f"{pad}{t.__name__}: {r}"

def inspect_checkpoint(path, show_values=False):
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"ERROR loading {path}: {e}", file=sys.stderr)
        return 2

    print(f"Loaded object type: {type(ckpt).__name__}")
    # common pattern: state_dict inside checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        print("Top-level keys:", list(ckpt.keys()))
        print("\n--- state_dict summary ---")
        sd = ckpt["state_dict"]
        for k,v in (sd.items() if show_values else list(sd.items())[:50]):
            if isinstance(v, torch.Tensor):
                print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"{k}: {type(v).__name__}")
        if not show_values and len(sd) > 50:
            print(f"... ({len(sd)-50} more keys)")
        return 0

    # otherwise, recursively summarize
    print(summarize(ckpt))
    return 0

def main():
    p = argparse.ArgumentParser(description="Inspect .pth / checkpoint file contents (safe: map to cpu).")
    p.add_argument("pth", help=".pth or checkpoint file path")
    p.add_argument("--show-values", action="store_true", help="print small non-tensor values and more keys (could be large)")
    args = p.parse_args()
    code = inspect_checkpoint(args.pth, show_values=args.show_values)
    sys.exit(code)

if __name__ == "__main__":
    main()