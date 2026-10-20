import ast
import logfire 
from langchain_core.documents import Document
 
SYMBOL_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
 
@logfire.instrument("Chunk Python files")
def chunk_python_document(document, max_chars=1000):
    """Split one Python file (a Document) into chunks.
 
    Returns None if the file cannot be parsed, so the caller can use the normal splitter.
    """
    source = document.page_content
 
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
 
    lines = source.splitlines()
    path = document.metadata.get("source", "unknown file")
 
    # 1. find the pieces of the file: functions, methods, classes and the code between them
    pieces = []
    collect_pieces(tree.body, lines, 1, len(lines), None, max_chars, pieces)
 
    # 2. turn every piece into one or more chunks (a piece that is too long is cut by lines)
    chunks = []
    for kind, name, class_name, start, end, docstring in pieces:
        groups = group_lines(lines, start, end, max_chars)
 
        for part, (first, last) in enumerate(groups, start=1):
            header = make_header(path, kind, name, class_name, first, last, docstring, part, len(groups))
            code = "\n".join(lines[first - 1:last])
 
            metadata = dict(document.metadata)
            metadata["symbol_type"] = kind            # module, class, function or method
            metadata["symbol_name"] = name
            metadata["class_name"] = class_name
            metadata["start_line"] = first
            metadata["end_line"] = last
 
            chunks.append(Document(page_content=f"{header}\n\n{code}", metadata=metadata))
    logfire.debug(
        "Python AST chunking completed",
        source_path=path,
        pieces_created=len(pieces),
        chunks_created=len(chunks),
        max_chars=max_chars,
    )

    return chunks
 
 
def collect_pieces(nodes, lines, first_line, last_line, class_name, max_chars, pieces):
    """Walk one block (the whole file, or the body of a big class) and add pieces in file order.
 
    A piece is a tuple: (kind, name, class_name, start_line, end_line, docstring)
    """
    cursor = first_line   # the next line that is not used yet
 
    for node in nodes:
        if not isinstance(node, SYMBOL_NODES):
            continue
 
        start = symbol_start(node, lines, cursor)
        end = node.end_lineno
 
        add_gap(pieces, lines, cursor, start - 1, class_name)   # code before this symbol
 
        if isinstance(node, ast.ClassDef):
            size = len("\n".join(lines[start - 1:end]))
            if size > max_chars:
                # big class: its header and attributes first, then one piece per method
                collect_pieces(node.body, lines, start, end, node.name, max_chars, pieces)
            else:
                pieces.append(("class", node.name, node.name, start, end, first_doc_line(node)))
        else:
            kind = "method" if class_name else "function"
            pieces.append((kind, node.name, class_name, start, end, first_doc_line(node)))
 
        cursor = end + 1
 
    add_gap(pieces, lines, cursor, last_line, class_name)       # code after the last symbol
 
 
def symbol_start(node, lines, cursor):
    """First line of a function or class: its decorators and the comment lines right above it."""
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
 
    while start > max(cursor, 1) and lines[start - 2].strip().startswith("#"):
        start -= 1
 
    return start
 
 
def add_gap(pieces, lines, first, last, class_name):
    """Add the lines that are not inside a function or class (imports, constants, scripts, comments)."""
    # ignore blank lines at both ends
    while first <= last and not lines[first - 1].strip():
        first += 1
    while last >= first and not lines[last - 1].strip():
        last -= 1
 
    if last < first:
        return
 
    if class_name:
        pieces.append(("class", class_name, class_name, first, last, None))
    else:
        pieces.append(("module", None, None, first, last, None))
 
 
def first_doc_line(node):
    """First line of the docstring, or None."""
    docstring = ast.get_docstring(node)
    if not docstring:
        return None
    return docstring.strip().split("\n")[0][:200]
 
 
def group_lines(lines, first, last, max_chars):
    """Split lines first..last (1-based, inclusive) into groups of at most max_chars characters."""
    groups = []
    group_start = first
    size = 0
 
    for number in range(first, last + 1):
        line_size = len(lines[number - 1]) + 1
 
        if size + line_size > max_chars and number > group_start:
            groups.append((group_start, number - 1))
            group_start = number
            size = 0
 
        size += line_size
 
    groups.append((group_start, last))
    return groups
 
 
def make_header(path, kind, name, class_name, first, last, docstring, part, parts):
    """The header that is put on top of every chunk."""
    header = [f"File: {path}"]
 
    if kind == "module":
        header.append("Module-level code (imports, constants, scripts)")
    if class_name:
        header.append(f"Class: {class_name}")
    if kind in ("function", "method"):
        header.append(f"Function: {name}")
 
    header.append(f"Lines: {first}-{last}")
 
    if parts > 1:
        header.append(f"Part: {part}/{parts}")
    if docstring:
        header.append(f"Docstring: {docstring}")
 
    return "\n".join(header)
