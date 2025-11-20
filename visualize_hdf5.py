import argparse
import h5py
import numpy as np
import sys


def summarize_dataset(dataset, max_preview=5, show_large_data=False):
    """Summarize an HDF5 dataset."""
    shape = dataset.shape
    dtype = dataset.dtype
    size_mb = dataset.size * dataset.dtype.itemsize / (1024 * 1024)
    
    info = f"Dataset: shape={shape}, dtype={dtype}, size={size_mb:.2f} MB"
    
    # Preview data based on size
    if dataset.size > 0:
        try:
            data = dataset[...]
            
            # Small datasets: show all
            if dataset.size <= 100:
                info += f"\n        values: {data}"
            
            # Medium datasets: show first few rows
            elif len(shape) == 1 and shape[0] <= max_preview * 20:
                info += f"\n        values: {data}"
            
            # Large 1D arrays
            elif len(shape) == 1:
                info += f"\n        first {max_preview} values: {data[:max_preview]}"
                info += f"\n        last {max_preview} values: {data[-max_preview:]}"
                info += f"\n        mean: {data.mean():.6f}, std: {data.std():.6f}"
                info += f"\n        min: {data.min():.6f}, max: {data.max():.6f}"
            
            # Large 2D arrays (matrices)
            elif len(shape) == 2:
                info += f"\n        first {max_preview}x{max_preview} corner:"
                corner = data[:max_preview, :max_preview]
                for row in corner:
                    info += f"\n          {row}"
                info += f"\n        statistics:"
                info += f"\n          mean: {data.mean():.6f}, std: {data.std():.6f}"
                info += f"\n          min: {data.min():.6f}, max: {data.max():.6f}"
                nonzero = (data != 0).sum()
                sparsity = 1 - (nonzero / data.size)
                info += f"\n          non-zero: {nonzero}/{data.size} ({100*(1-sparsity):.1f}%)"
                if sparsity > 0.5:
                    info += f"\n          sparsity: {sparsity*100:.1f}%"
            
            # Higher dimensional arrays
            else:
                info += f"\n        shape: {shape}"
                info += f"\n        mean: {data.mean():.6f}, std: {data.std():.6f}"
                info += f"\n          min: {data.min():.6f}, max: {data.max():.6f}"
                
        except Exception as e:
            info += f"\n        (could not preview: {e})"
    
    return info


def summarize_attrs(attrs, indent=8):
    """Summarize attributes."""
    if not attrs:
        return ""
    
    pad = " " * indent
    lines = []
    for key, value in attrs.items():
        if isinstance(value, (np.ndarray, list)):
            if isinstance(value, np.ndarray):
                val_str = f"ndarray shape={value.shape}, dtype={value.dtype}"
            else:
                val_str = f"list len={len(value)}"
            lines.append(f"{pad}@{key}: {val_str}")
        elif isinstance(value, bytes):
            try:
                decoded = value.decode('utf-8')
                if len(decoded) > 100:
                    decoded = decoded[:100] + "..."
                lines.append(f"{pad}@{key}: {repr(decoded)}")
            except:
                lines.append(f"{pad}@{key}: bytes[{len(value)}]")
        else:
            val_str = repr(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            lines.append(f"{pad}@{key}: {val_str}")
    
    return "\n".join(lines)


def traverse_hdf5(group, indent=0, max_depth=None, show_attrs=True):
    """Recursively traverse and print HDF5 group structure."""
    pad = "  " * indent
    
    if max_depth is not None and indent >= max_depth:
        print(f"{pad}... (max depth reached)")
        return
    
    # Group attributes
    if show_attrs and len(group.attrs) > 0:
        print(f"{pad}Attributes:")
        print(summarize_attrs(group.attrs, indent=(indent+1)*2))
    
    # Iterate through items
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"{pad}📁 Group: {name}/")
            traverse_hdf5(item, indent + 1, max_depth, show_attrs)
        
        elif isinstance(item, h5py.Dataset):
            print(f"{pad}📄 {name}")
            print(f"{pad}    {summarize_dataset(item)}")
            
            if show_attrs and len(item.attrs) > 0:
                print(f"{pad}    Attributes:")
                print(summarize_attrs(item.attrs, indent=(indent+2)*2))


def get_file_info(f):
    """Get basic file information."""
    info = []
    info.append(f"HDF5 File Information:")
    info.append(f"  Driver: {f.driver}")
    info.append(f"  Mode: {f.mode}")
    
    # Count groups and datasets
    n_groups = 0
    n_datasets = 0
    
    def count_items(group):
        nonlocal n_groups, n_datasets
        for item in group.values():
            if isinstance(item, h5py.Group):
                n_groups += 1
                count_items(item)
            elif isinstance(item, h5py.Dataset):
                n_datasets += 1
    
    count_items(f)
    info.append(f"  Total Groups: {n_groups}")
    info.append(f"  Total Datasets: {n_datasets}")
    
    return "\n".join(info)


def list_datasets(f, prefix=""):
    """List all datasets with their paths."""
    datasets = []
    
    def collect(group, path):
        for name, item in group.items():
            full_path = f"{path}/{name}"
            if isinstance(item, h5py.Group):
                collect(item, full_path)
            elif isinstance(item, h5py.Dataset):
                datasets.append((full_path, item.shape, item.dtype))
    
    collect(f, prefix)
    return datasets


def inspect_hdf5(path, max_depth=None, show_attrs=True, list_only=False, dataset_path=None):
    """Inspect HDF5 file structure."""
    try:
        with h5py.File(path, 'r') as f:
            print("=" * 80)
            print(f"File: {path}")
            print("=" * 80)
            print()
            
            # File info
            print(get_file_info(f))
            print()
            
            if list_only:
                # List all datasets
                print("All Datasets:")
                print("-" * 80)
                datasets = list_datasets(f)
                for path, shape, dtype in datasets:
                    print(f"{path:60s} shape={str(shape):20s} dtype={dtype}")
                return 0
            
            if dataset_path:
                # Show specific dataset
                if dataset_path in f:
                    item = f[dataset_path]
                    if isinstance(item, h5py.Dataset):
                        print(f"Dataset: {dataset_path}")
                        print("-" * 80)
                        print(summarize_dataset(item))
                        if show_attrs and len(item.attrs) > 0:
                            print("\nAttributes:")
                            print(summarize_attrs(item.attrs, indent=2))
                        
                        # Show actual data for small datasets
                        if item.size <= 1000:
                            print("\nData:")
                            data = item[...]
                            print(data)
                    else:
                        print(f"Error: {dataset_path} is not a dataset")
                        return 1
                else:
                    print(f"Error: {dataset_path} not found in file")
                    return 1
                return 0
            
            # Tree structure
            print("File Structure:")
            print("-" * 80)
            traverse_hdf5(f, indent=0, max_depth=max_depth, show_attrs=show_attrs)
            
        return 0
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


def main():
    parser = argparse.ArgumentParser(
        description="Inspect HDF5 file structure and contents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show full tree structure
  python inspect_hdf5.py data.hdf5
  
  # Limit depth to 2 levels
  python inspect_hdf5.py data.hdf5 --max-depth 2
  
  # List all datasets only
  python inspect_hdf5.py data.hdf5 --list
  
  # Show specific dataset
  python inspect_hdf5.py data.hdf5 --dataset /data/obs/object
  
  # Hide attributes
  python inspect_hdf5.py data.hdf5 --no-attrs
        """
    )
    parser.add_argument("hdf5", help="HDF5 file path")
    parser.add_argument("--max-depth", type=int, default=None,
                       help="Maximum depth to traverse (default: unlimited)")
    parser.add_argument("--no-attrs", action="store_true",
                       help="Don't show attributes")
    parser.add_argument("--list", action="store_true",
                       help="List all datasets with paths only")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Show specific dataset by path (e.g., /data/demo_0/actions)")
    
    args = parser.parse_args()
    
    code = inspect_hdf5(
        args.hdf5,
        max_depth=args.max_depth,
        show_attrs=not args.no_attrs,
        list_only=args.list,
        dataset_path=args.dataset
    )
    sys.exit(code)


if __name__ == "__main__":
    main()