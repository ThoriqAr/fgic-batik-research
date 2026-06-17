from pathlib import Path

IGNORE_DIRS = {
    "__pycache__",
    ".venv",
    ".git",
    "attribution_maps",
    "batik_betawi",
    "batik_bokor_kencono",
    "batik_buketan",
    "batik_dayak",
    "batik_jlamprang",
    "batik_kawung",
    "batik_liong",
    "batik_mega_mendung",
    "batik_parang",
    "batik_sekarjagad",
    "batik_sidoluhur",
    "batik_sidomukti",
    "batik_sidomulyo",
    "batik_singa_barong",
    "batik_srikaton",
    "batik_tribusono",
    "batik_tujuh_rupa",
    "batik_tuntrum",
    "batik_wahyu_tumurun",
    "batik_wirasat"
}

IGNORE_FILES = {
    "__init__.py"
}

def print_tree(path, prefix=""):

    items = sorted(path.iterdir())

    items = [
        p for p in items
        if p.name not in IGNORE_DIRS
        and p.name not in IGNORE_FILES
    ]

    for i, item in enumerate(items):

        connector = "└── " if i == len(items)-1 else "├── "

        print(prefix + connector + item.name)

        if item.is_dir():

            extension = "    " if i == len(items)-1 else "│   "

            print_tree(
                item,
                prefix + extension
            )

print(".")
print_tree(Path("."))
