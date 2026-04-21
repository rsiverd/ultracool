import ctypes
import gc

## Recursively gather getsizeof() for things ...
def get_ids_recursive(thing):
    return [id(x) for x in gc.get_referents(thing)]

def object_from_id(objid):
    return ctypes.cast(objid, ctypes.py_object).value

def size_from_id(objid):
    return sys.getsizeof(ctypes.cast(objid, ctypes.py_object).value)

def recursively_walk_object(thing):
    return gc.get_referents(thing)

#asdf = get_ids_recursive(all_data)

## ----------------------------------------------------------------------- 

def get_all_referents(obj):
    """Recursively finds all objects referred to by the input object."""
    seen = {id(obj)}
    to_process = [obj]
    all_referents = []

    while to_process:
        current = to_process.pop()
        # Get immediate referents
        referents = gc.get_referents(current)

        for ref in referents:
            if id(ref) not in seen:
                seen.add(id(ref))
                all_referents.append(ref)
                to_process.append(ref)

    return all_referents

## ----------------------------------------------------------------------- 

def get_all_referents_and_sizes(obj):
    """Recursively finds all objects referred to by the input object."""
    seen = {id(obj)}
    to_process = [obj]
    all_referents = []
    all_bytesizes = []

    while to_process:
        current = to_process.pop()
        # Get immediate referents
        referents = gc.get_referents(current)

        for ref in referents:
            if id(ref) not in seen:
                seen.add(id(ref))
                all_bytesizes.append(sys.getsizeof(ref))
                all_referents.append(ref)
                to_process.append(ref)

    return all_referents, all_bytesizes

def sizeprint(sizes_iterable, stream=sys.stdout):
    total_count = len(sizes_iterable)
    total_bytes = np.sum(list(sizes_iterable))
    total_kib   = total_bytes / 1024.
    total_mib   = total_kib / 1024.
    total_gib   = total_mib / 1024.
    stream.write("Have %d items. Total size:\n" % total_count)
    stream.write("-->  %d bytes\n" % total_bytes)
    stream.write("-->  %.1f KiB\n" % total_kib)
    stream.write("-->  %.1f MiB\n" % total_mib)
    stream.write("-->  %.1f GiB\n" % total_gib)
    return total_bytes

def easy_sizeprint(obj):
    _refs, _sizes = get_all_referents_and_sizes(obj)
    sizeprint(_sizes)

# Example Usage:
all_refs = get_all_referents(data)
print(f"Found {len(all_refs)} unique referents.")

