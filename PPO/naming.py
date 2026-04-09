"""
Helper just for naming for plot titles and tables
"""

def fmt_decimal(token):
    token = token.strip().lower()
    if not token.isdigit():
        return token
    if token == "10":
        return "1.0"
    if len(token) == 3 and token.startswith("0"):
        return f"0.{token[1:]}"
    if len(token) == 1:
        return f"0.{token}"
    return token
 
 
def pretty_run_name(run_name):
    if not run_name:
        return "Baseline"
    parts = run_name.split("_")
    out = []
    i = 0
    while i < len(parts):
        p = parts[i].lower()
        if p == "baseline":
            out.append("Baseline")
        elif p.startswith("lr") and len(p) > 2:
            out.append(f"lr = {p[2:]}")
        elif p == "gae" and i + 1 < len(parts):
            out.append(f"GAE = {fmt_decimal(parts[i + 1])}")
            i += 1
        elif p.startswith("ent") and len(p) > 3:
            out.append(f"ent = {fmt_decimal(p[3:])}")
        elif p == "ent" and i + 1 < len(parts):
            out.append(f"ent = {fmt_decimal(parts[i + 1])}")
            i += 1
        elif p == "vf" and i + 2 < len(parts) and parts[i + 1].lower() == "coef":
            out.append(f"vf coef = {fmt_decimal(parts[i + 2])}")
            i += 2
        elif p == "manhattan" and i + 1 < len(parts):
            out.append(f"Manhattan = {fmt_decimal(parts[i + 1])}")
            i += 1
        elif p.endswith("k") and p[:-1].isdigit():
            out.append(p)
        elif p == "5seeds":
            out.append("verification")
        else:
            out.append(parts[i])
        i += 1
    return ", ".join(out)