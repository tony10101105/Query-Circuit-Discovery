"""
Refactoring script for EAP-IG analysis files.
Run: python3 _refactor.py  (from the EAP-IG/ directory)
"""
import os, re

EAPIG_DIR = "/home/tony/Query-Circuit-Discovery/EAP-IG"

# ─── low-level helpers ──────────────────────────────────────────────────────

def remove_top_level_block(content, name, kind="def"):
    """Remove a top-level function or class definition, handling multi-line signatures."""
    pattern = rf'^{kind} {re.escape(name)}\b'
    lines = content.split('\n')
    result, i = [], 0
    while i < len(lines):
        if re.match(pattern, lines[i]):
            i += 1
            while i < len(lines):
                line = lines[i]
                # Stop only at genuine new col-0 statement (not ), ], etc.)
                if line and not line[0].isspace() and re.match(r'^[a-zA-Z_@#"\']', line):
                    break
                i += 1
            while result and result[-1].strip() == '':
                result.pop()
        else:
            result.append(lines[i]); i += 1
    return '\n'.join(result)

def remove_exact_line(content, pattern):
    return re.sub(r'^' + pattern + r'[ \t]*\n', '', content, flags=re.MULTILINE)

def fix_score_paths(content):
    content = re.sub(r'"score_data//', '"Query-Circuit-Dataset/score_data/', content)
    content = re.sub(r'"score_data/', '"Query-Circuit-Dataset/score_data/', content)
    content = re.sub(r"'score_data//", "'Query-Circuit-Dataset/score_data/", content)
    content = re.sub(r"'score_data/", "'Query-Circuit-Dataset/score_data/", content)
    return content

def set_v4(content):
    return re.sub(r'\bmetric_version\s*=\s*8\b', 'metric_version = 4', content)

def remove_version_kwarg(content):
    return re.sub(r',\s*version=metric_version\b', '', content)

def remove_local_set_seed(content):
    return remove_top_level_block(content, 'set_seed', 'def')

def insert_after_last_import(content, import_line):
    lines = content.split('\n')
    last = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('import ') or s.startswith('from '):
            last = i
    if last < 0: lines.insert(0, import_line)
    else: lines.insert(last + 1, import_line)
    return '\n'.join(lines)

def ensure_set_seed_import(content):
    """Add set_seed to src.eap.utils import chain, or add a new line."""
    if re.search(r'from src\.eap\.utils import', content):
        def add_ss(m):
            return m.group(0) if 'set_seed' in m.group(0) else m.group(0).rstrip() + ', set_seed'
        return re.sub(r'from src\.eap\.utils import [^\n]+', add_ss, content, count=1)
    lines = content.split('\n')
    last_eap = max((i for i, l in enumerate(lines) if l.strip().startswith('from src.eap.')), default=-1)
    if last_eap >= 0:
        lines.insert(last_eap + 1, 'from src.eap.utils import set_seed')
    return '\n'.join(lines)

def remove_eap_defs(content, is_para=False):
    if is_para:
        content = remove_top_level_block(content, 'PARA_collate_EAP', 'def')
        content = remove_top_level_block(content, 'PARA_EAPDataset', 'class')
    else:
        content = remove_top_level_block(content, 'collate_EAP', 'def')
        content = remove_top_level_block(content, 'EAPDataset', 'class')
    content = remove_top_level_block(content, 'get_logit_positions', 'def')
    content = remove_top_level_block(content, 'logit_diff', 'def')
    return content

# ─── import strings ──────────────────────────────────────────────────────────
SI  = "from utils import get_logit_positions, logit_diff_simple as logit_diff, EAPDataset, collate_EAP"
SPI = "from utils import get_logit_positions, logit_diff_simple as logit_diff, SimplePARAEAPDataset as EAPDataset, collate_EAP_mc as collate_EAP"
MCI = "from utils import get_logit_positions, logit_diff_mc as logit_diff, MCEAPDataset as EAPDataset, collate_EAP_mc as collate_EAP"
PI  = "from utils import get_logit_positions, logit_diff_mc as logit_diff, PARAEAPDataset as PARA_EAPDataset, collate_para as PARA_collate_EAP"

# ─── high-level transforms ───────────────────────────────────────────────────

def transform_simple(c, subp=False, env=True, lss=False, scores=True):
    if subp:
        c = remove_exact_line(c, r'import subprocess')
        c = remove_exact_line(c, r'subprocess\.Popen\(\["python3", "gpu_keepalive\.py"\]\)')
    if env:  c = remove_exact_line(c, r'os\.environ\["TRANSFORMERS_CACHE"\] = "[^"]*"')
    if lss:  c = remove_local_set_seed(c); c = ensure_set_seed_import(c)
    c = remove_eap_defs(c)
    c = insert_after_last_import(c, SI)
    if scores: c = fix_score_paths(c)
    return c

def transform_simple_para(c, env=True):
    if env:  c = remove_exact_line(c, r'os\.environ\["TRANSFORMERS_CACHE"\] = "[^"]*"')
    c = remove_eap_defs(c)
    c = insert_after_last_import(c, SPI)
    return c

def transform_mc(c, env=True, lss=False, scores=True, upd_v4=False):
    if env:  c = remove_exact_line(c, r'os\.environ\["TRANSFORMERS_CACHE"\] = "[^"]*"')
    if lss:  c = remove_local_set_seed(c); c = ensure_set_seed_import(c)
    c = remove_eap_defs(c)
    c = insert_after_last_import(c, MCI)
    if scores:  c = fix_score_paths(c)
    if upd_v4:  c = set_v4(c); c = remove_version_kwarg(c)
    return c

def transform_para(c, env=True):
    if env:  c = remove_exact_line(c, r'os\.environ\["TRANSFORMERS_CACHE"\] = "[^"]*"')
    c = remove_eap_defs(c, is_para=True)
    c = insert_after_last_import(c, PI)
    c = remove_version_kwarg(c)
    return c

def transform_score_only(c):
    c = remove_exact_line(c, r'os\.environ\["TRANSFORMERS_CACHE"\] = "[^"]*"')
    c = fix_score_paths(c)
    return c

# ─── runner ──────────────────────────────────────────────────────────────────

def process(fname, fn, **kw):
    path = os.path.join(EAPIG_DIR, fname)
    orig = open(path).read()
    new  = fn(orig, **kw)
    if new != orig:
        open(path,'w').write(new); print(f"  [CHANGED]   {fname}")
    else:
        print(f"  [UNCHANGED] {fname}")

print("=== Transforming EAP-IG analysis files ===\n")

process("appendix_score_matrix.py",        transform_score_only)
process("arcc_analysis.py",                transform_mc)
process("arcc_analysis_complement.py",     transform_mc)
process("arcc_one_sample_paraphrase.py",   transform_para)
process("arithmeticadd_analysis.py",           transform_simple)
process("arithmeticadd_analysis_complement.py",transform_simple)
process("arithmeticadd_one_sample_paraphrase.py", transform_simple_para)
process("arithmeticmul_analysis.py",           transform_simple, subp=True)
process("arithmeticmul_analysis_complement.py",transform_simple)
process("arithmeticmul_analysis_para_num.py",  transform_simple)
process("arithmeticmul_analysis_perturb.py",   transform_simple)
process("arithmeticmul_one_sample_paraphrase.py", transform_simple_para)
process("figure3_nfs_ndf_comp_mmlu_one_sample.py", transform_mc, upd_v4=True)
process("figure4_ioi_analysis.py",             transform_mc,  upd_v4=True)
process("gender_bias_greedy_dijkstra_comp.py", transform_simple, env=False, scores=False)
process("gender_bias_one_sample.py",           transform_simple, env=False, lss=True, scores=False)
process("ioi_analysis.py",                     transform_simple, subp=True)
process("ioi_analysis_perturb.py",             transform_simple)
process("ioi_analysis_para_num.py",            transform_simple)
process("ioi_canonical.py",                    transform_simple, env=False, scores=False)
process("ioi_canonical_one_sample.py",         transform_simple, env=False, lss=True, scores=False)
process("ioi_one_sample.py",                   transform_simple, env=False)
process("mmlu_analysis.py",                    transform_mc)
process("mmlu_analysis_complement.py",         transform_mc)
process("mmlu_analysis_para_num.py",           transform_mc)
process("mmlu_analysis_perturb.py",            transform_mc)
process("mmlu_one_sample_paraphrase.py",       transform_para)

print("\n=== Post-processing: gender_bias column name fixes ===\n")
for fname, extra in [("gender_bias_greedy_dijkstra_comp.py",""),
                     ("gender_bias_one_sample.py",", data_num=data_num")]:
    path = os.path.join(EAPIG_DIR, fname)
    orig = open(path).read()
    old = f"EAPDataset('probing_dataset/gender_bias_gpt2.csv'{extra})"
    new_call = (f"EAPDataset('probing_dataset/gender_bias_gpt2.csv'{extra}, "
                f"correct_col='clean_answer_idx', incorrect_col='corrupted_answer_idx')")
    c = orig.replace(old, new_call)
    if c != orig:
        open(path,'w').write(c); print(f"  [FIXED cols] {fname}")
    else:
        print(f"  [no change]  {fname}")

print("\nAll done.")
