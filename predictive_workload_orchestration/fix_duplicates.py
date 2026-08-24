"""Remove the duplicate sections 2 and 3 from PROJECT_FILES.md."""
with open("PROJECT_FILES.md", "r") as f:
    lines = f.readlines()

# Find both occurrences of "## 2. config/"
occurrences = [i for i, line in enumerate(lines) if line.strip() == "## 2. config/"]
print("Occurrences at lines:", occurrences)

if len(occurrences) >= 2:
    first = occurrences[0]
    second = occurrences[1]
    # Walk back from first occurrence to find the preceding "---" separator
    cut_start = first
    for i in range(first - 1, max(first - 5, 0), -1):
        if lines[i].strip() == "---":
            cut_start = i
            break
    print(f"Removing lines {cut_start} to {second - 2} (0-indexed, inclusive)")
    # Keep: everything before cut_start, then from (second-1) onward
    clean = lines[:cut_start] + lines[second - 1:]
    with open("PROJECT_FILES.md", "w") as f:
        f.writelines(clean)
    print(f"Done. Lines before: {len(lines)}  Lines after: {len(clean)}")
else:
    print("No duplicate found.")
