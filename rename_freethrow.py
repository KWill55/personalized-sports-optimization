from pathlib import Path

for file in Path(".").rglob("*Freethrow*"):
    new_name = file.name.replace("Freethrow", "freethrow")
    new_path = file.with_name(new_name)
    if file.name != new_path.name:
        print(f"Renaming {file} → {new_path}")
        file.rename(new_path)
