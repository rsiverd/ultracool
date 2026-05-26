import ctypes
import sys
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

# from:
# https://stackoverflow.com/questions/13530762/how-to-know-bytes-size-of-python-object-like-arrays-and-dictionaries-the-simp
def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id:o for o_id,o in all_refr \
                if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return marked,sz


def sizeprint(sizes_iterable, stream=sys.stdout):
    total_count = len(sizes_iterable)
    total_bytes = sum(list(sizes_iterable))
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
#all_refs = get_all_referents(data)
#print(f"Found {len(all_refs)} unique referents.")

