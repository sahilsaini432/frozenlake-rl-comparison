def format_decimal_code(token):
    token = token.strip().lower()

    if token.isdigit():
        if len(token) == 3:
            return f"0.{token[1:]}"
        if len(token) == 2:
            return f"0.{token[1]}"
        if len(token) == 1:
            return f"0.{token}"

    return token


def pretty_run_name(run_name):
    if not run_name:
        return "Baseline"

    parts = run_name.split("_")
    pretty_parts = []

    i = 0
    while i < len(parts):
        part = parts[i].lower()

        if part == "baseline":
            pretty_parts.append("Baseline")

        elif part.startswith("lr") and len(part) > 2:
            pretty_parts.append(f"lr = {part[2:]}")

        elif part == "gae" and i + 1 < len(parts):
            pretty_parts.append(f"GAE = {format_decimal_code(parts[i + 1])}")
            i += 1

        elif part.startswith("ent") and len(part) > 3:
            pretty_parts.append(f"ent = {format_decimal_code(part[3:])}")

        elif part == "ent" and i + 1 < len(parts):
            pretty_parts.append(f"ent = {format_decimal_code(parts[i + 1])}")
            i += 1

        elif part == "vf" and i + 2 < len(parts) and parts[i + 1].lower() == "coef":
            pretty_parts.append(f"vf coef = {format_decimal_code(parts[i + 2])}")
            i += 2

        elif part == "manhattan" and i + 1 < len(parts):
            pretty_parts.append(f"Manhattan = {format_decimal_code(parts[i + 1])}")
            i += 1

        elif part.endswith("k") and part[:-1].isdigit():
            pretty_parts.append(part)

        elif part == "5seeds":
            pretty_parts.append("verification")

        else:
            pretty_parts.append(parts[i])

        i += 1

    return ", ".join(pretty_parts)