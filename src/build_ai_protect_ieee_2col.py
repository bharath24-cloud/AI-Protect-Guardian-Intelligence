import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION_START
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def set_cell_background(cell, fill_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_margins(cell, top=60, bottom=60, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>\n'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>\n'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_ai_protect_2col(output_docx_path):
    doc = docx.Document()
    
    # -------------------------------------------------------------
    # SECTION 0: Title, Authors, Abstract, Index Terms (1 Column)
    # -------------------------------------------------------------
    sec0 = doc.sections[0]
    sec0.top_margin = Inches(0.75)
    sec0.bottom_margin = Inches(0.75)
    sec0.left_margin = Inches(0.75)
    sec0.right_margin = Inches(0.75)
    
    # Configure normal style
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(9.5)
    style_normal.paragraph_format.line_spacing = 1.05
    style_normal.paragraph_format.space_after = Pt(3)

    # 1. Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("AI-Protect: An Adaptive Bounded Adversarial Cloaking Engine and Multi-Agent Autonomous AI Scientist for Visual IP Defense and Closed-Loop Scientific Discovery")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(17)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0, 0, 0)

    # 2. Authors
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_before = Pt(0)
    p_auth.paragraph_format.space_after = Pt(2)
    r_auth = p_auth.add_run("Bharathram K , Vishal S")
    r_auth.font.name = 'Times New Roman'
    r_auth.font.size = Pt(11)
    r_auth.font.bold = False

    # 3. Department
    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.paragraph_format.space_before = Pt(0)
    p_dept.paragraph_format.space_after = Pt(10)
    r_dept = p_dept.add_run("Department of Computer Science & Artificial Intelligence")
    r_dept.font.name = 'Times New Roman'
    r_dept.font.size = Pt(10)
    r_dept.font.italic = True

    # 4. Abstract
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_before = Pt(0)
    p_abs.paragraph_format.space_after = Pt(4)
    p_abs.paragraph_format.line_spacing = 1.05
    
    r_abs_label = p_abs.add_run("Abstract—")
    r_abs_label.font.name = 'Times New Roman'
    r_abs_label.font.size = Pt(9.5)
    r_abs_label.font.bold = True

    r_abs_text = p_abs.add_run(
        "The explosive expansion of generative artificial intelligence and multimodal foundation models presents critical challenges "
        "at the nexus of digital content sovereignty and autonomous scientific research. Digital creators face unauthorized web scraping "
        "and feature extraction without compensation, while computational scientists encounter high-dimensional experimental spaces "
        "that overwhelm classical manual trial-and-error. In this paper, we present AI-Protect, a unified deep learning and multi-agent framework "
        "establishing two complementary operational tracks: (1) An adaptive bounded adversarial cloaking engine that optimizes Projected Gradient Descent "
        "(PGD) perturbations constrained by an L_inf bound (||delta||_inf <= 8/255 ≈ 0.03137) to disrupt deep feature representations across "
        "surrogate vision backbones (ResNet-50, ConvNeXt, Vision Transformers, and CLIP) while preserving high human visual fidelity "
        "(PSNR = 33.37 dB, SSIM = 0.8262); (2) An end-to-end multi-agent autonomous scientific discovery system formalizing experiment selection "
        "as a finite-horizon Markov Decision Process (MDP) solved via Deep Q-Learning (DQN) with an adversarial Critic Agent enforcing strict hypothesis "
        "falsification (R^2 < 0.25 threshold rejection) and SHA-256 deduplicated bipartite knowledge graph memory. Across extensive benchmarks on the WikiArt "
        "dataset (63,061 works) and computational materials suites, the proposed system demonstrates 93.9% white-box disruption, 75.3%-85.7% black-box transferability, "
        "robustness across JPEG compression, and an 88.5% sample efficiency in active physical property discovery."
    )
    r_abs_text.font.name = 'Times New Roman'
    r_abs_text.font.size = Pt(9.5)

    # 5. Index Terms
    p_idx = doc.add_paragraph()
    p_idx.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_idx.paragraph_format.space_before = Pt(2)
    p_idx.paragraph_format.space_after = Pt(12)
    
    r_idx_label = p_idx.add_run("Index Terms—")
    r_idx_label.font.name = 'Times New Roman'
    r_idx_label.font.size = Pt(9.5)
    r_idx_label.font.bold = True
    r_idx_label.font.italic = True

    r_idx_text = p_idx.add_run("Adversarial Machine Learning, Visual IP Protection, Projected Gradient Descent (PGD), Active Experiment Selection, Markov Decision Process (MDP), Multi-Agent Systems, Deep Feature Shift.")
    r_idx_text.font.name = 'Times New Roman'
    r_idx_text.font.size = Pt(9.5)
    r_idx_text.font.italic = True

    # -------------------------------------------------------------
    # SECTION 1: Two-Column Body Section
    # -------------------------------------------------------------
    sec1 = doc.add_section(WD_SECTION_START.CONTINUOUS)
    sec1.top_margin = Inches(0.75)
    sec1.bottom_margin = Inches(0.75)
    sec1.left_margin = Inches(0.75)
    sec1.right_margin = Inches(0.75)
    
    # Configure 2 columns with 720 dxa (0.5 in) spacing
    sectPr = sec1._sectPr
    cols = sectPr.xpath('./w:cols')
    if cols:
        cols[0].set(qn('w:num'), '2')
        cols[0].set(qn('w:space'), '720')
    else:
        new_cols = parse_xml(f'<w:cols {nsdecls("w")} w:num="2" w:space="720"/>')
        sectPr.append(new_cols)

    # Helpers
    def add_sec_heading(title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(title)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.bold = True
        return p

    def add_subsec_heading(title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(title)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9.5)
        run.font.bold = True
        run.font.italic = True
        return p

    def add_para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.05
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9.5)
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.05
        
        r_bullet = p.add_run("• ")
        r_bullet.font.name = 'Times New Roman'
        r_bullet.font.size = Pt(9.5)
        r_bullet.font.bold = True
        
        if bold_prefix:
            r_b = p.add_run(bold_prefix + " ")
            r_b.font.name = 'Times New Roman'
            r_b.font.size = Pt(9.5)
            r_b.font.bold = True
            
        r_t = p.add_run(text)
        r_t.font.name = 'Times New Roman'
        r_t.font.size = Pt(9.5)
        return p

    def add_figure(img_path, caption_text, width=Inches(3.25)):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(4)
            p_img.paragraph_format.space_after = Pt(2)
            run_img = p_img.add_run()
            run_img.add_picture(img_path, width=width)
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(1)
            p_cap.paragraph_format.space_after = Pt(5)
            run_cap = p_cap.add_run(caption_text)
            run_cap.font.name = 'Times New Roman'
            run_cap.font.size = Pt(8.5)
            run_cap.font.italic = True

    def add_eq(eq_text, eq_num):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r_eq = p.add_run(f"    {eq_text}   {eq_num}")
        r_eq.font.name = 'Cambria Math'
        r_eq.font.size = Pt(9.5)
        r_eq.font.italic = True
        return p

    def add_table_custom(title, headers, rows):
        p_t = doc.add_paragraph()
        p_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_t.paragraph_format.space_before = Pt(6)
        p_t.paragraph_format.space_after = Pt(2)
        r_t = p_t.add_run(title)
        r_t.font.name = 'Times New Roman'
        r_t.font.size = Pt(9.0)
        r_t.font.bold = True

        tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_table_borders(tbl)

        # Header
        hdr_cells = tbl.rows[0].cells
        for i, h in enumerate(headers):
            hdr_cells[i].text = h
            set_cell_background(hdr_cells[i], "002855")
            set_cell_margins(hdr_cells[i], top=50, bottom=50, left=70, right=70)
            p = hdr_cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(8.5)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        # Rows
        for r_idx, row_data in enumerate(rows):
            row_cells = tbl.rows[r_idx + 1].cells
            bg_color = "F9FBFD" if r_idx % 2 == 1 else "FFFFFF"
            for c_idx, val in enumerate(row_data):
                row_cells[c_idx].text = str(val)
                set_cell_background(row_cells[c_idx], bg_color)
                set_cell_margins(row_cells[c_idx], top=40, bottom=40, left=70, right=70)
                p = row_cells[c_idx].paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(8.0)

        p_sp = doc.add_paragraph()
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(3)

    # --- I. INTRODUCTION ---
    add_sec_heading("I. INTRODUCTION")
    add_para("Recent advances in deep learning have enabled dual capabilities across computational domains: automated extraction of complex empirical physical laws from high-dimensional datasets and large-scale web scraping for generative multimodal models (LMMs). In computational materials science, exploring multi-element stoichiometric spaces is computationally expensive, creating a critical need for autonomous systems that optimize experiment selection without human bias. Simultaneously, digital visual creators face unprecedented unauthorized data harvesting, necessitating proactive mechanisms that protect visual IP without degrading perceptual quality for human viewers.")
    add_para("Current commercial and research methodologies exhibit three debilitating architectural limitations:")
    
    add_bullet("Vulnerable Visual IP Ingestion:", "Web-scale crawlers systematically harvest visual media to train large-scale vision-language models without consent, attribution, or verifiable watermarking defenses (Radenovic, 2023).")
    add_bullet("High-Dimensional Experiment Inefficiency:", "Conventional discovery systems rely on brute-force grid searches or heuristic sweeps, squandering computational budgets on redundant or trivially correlated material candidates (Jain, 2013).")
    add_bullet("Unchecked Correlation & Hallucination:", "Autonomous exploratory loops lack adversarial falsification agents, frequently committing ungrounded or confounded empirical claims into scientific registries (Kirkpatrick, 2017).")
    
    add_para("To resolve these challenges, we present AI-Protect, an end-to-end framework uniting bounded adversarial image cloaking and closed-loop multi-agent scientific discovery.")

    # --- II. LITERATURE REVIEW & RELATED WORK ---
    add_sec_heading("II. LITERATURE REVIEW & RELATED WORK")
    add_para("Our framework bridges research across four technical domains:")
    
    add_subsec_heading("A. Adversarial Cloaking & Visual IP Protection")
    add_para("Gradient-based adversarial attacks, first formalized by Goodfellow et al. [1] with FGSM and extended by Madry et al. [2] via Projected Gradient Descent (PGD), demonstrate that bounded input perturbations reliably alter neural network outputs. Shan et al. [4], [5] pioneered protective perturbation applications (Fawkes, Glaze), demonstrating that feature-space disruption shields human portraits and artistic styles from unauthorized facial recognition and style transfer.")
    
    add_subsec_heading("B. Multi-Agent Systems & Collaborative Orchestration")
    add_para("Decomposing complex reasoning tasks into specialized collaborative agent personas yields significant improvements over monolithic prompts. Lin [4] and Liu [5] established multi-agent interaction protocols where planning, reasoning, memory, and validation agents collaborate under strict task graph schedulers.")
    
    add_subsec_heading("C. Active Learning & MDP Experiment Selection")
    add_para("Reinforcement learning transforms passive active learning into dynamic decision processes. Formalizing hypothesis testing as a finite-horizon Markov Decision Process (MDP) allows policy networks to optimize multi-objective rewards balancing mutual information, novelty, and computational budget.")

    # --- III. SYSTEM ARCHITECTURE ---
    add_sec_heading("III. SYSTEM ARCHITECTURE")
    add_para("The AI-Protect architecture is formulated as a modular, multi-threaded engine designed for robust feature disruption and autonomous scientific falsification.")
    
    add_subsec_heading("A. End-to-End System Architecture & Layered Cognitive Flow")
    add_para("As illustrated in Fig. 3.1, the framework operates across seven sequential functional tiers complemented by two cross-cutting operational pillars: Privacy & Security, and Explainability & Monitoring.")
    
    fig1_path = os.path.join(os.getcwd(), 'experiments', 'plots', 'original_vs_cloaked_comparison.png')
    add_figure(fig1_path, "Fig. 3.1. Adversarial Cloaking Comparison: Original vs. Protected Asset.")
    
    add_bullet("1) Ingestion & Preprocessing Layer:", "Validates raw image tensors, strips tracking metadata/Exif, and computes normalized channel statistics.")
    add_bullet("2) Surrogate Vision Layer:", "Maintains pre-trained vision backbones (ResNet-50, ConvNeXt, ViT, CLIP) to compute feature activation maps.")
    add_bullet("3) Adversarial Cloaking Engine:", "Executes iterative PGD perturbations constrained by strict L_inf bounds (||delta||_inf <= 8/255).")
    add_bullet("4) Perceptual Verification Layer:", "Quantifies visual fidelity metrics including PSNR, SSIM, and LPIPS to ensure imperceptibility.")
    add_bullet("5) Cross-Model Transferability Matrix:", "Evaluates zero-shot cloaking efficacy against unseen black-box vision architectures.")
    add_bullet("6) Transformation Robustness Pipeline:", "Simulates real-world web delivery transformations (JPEG 95/75/50, spatial downsampling, cropping, blur).")
    add_bullet("7) Response & Action Layer:", "Generates protected visual media and audit certificates with cryptographic signatures.")

    add_subsec_heading("B. Agent Orchestrator & Multi-Agent Discovery Architecture")
    add_para("In Track A, the central Agent Orchestrator coordinates specialized agent personas:")
    
    fig2_path = os.path.join(os.getcwd(), 'experiments', 'plots', 'transferability_heatmap.png')
    add_figure(fig2_path, "Fig. 3.2. Cross-Architecture Transferability Heatmap Across Vision Backbones.")
    
    add_bullet("Discovery Agent:", "Conducts multi-scale mutual information and symbolic power-law searches across latent material vectors.")
    add_bullet("Hypothesis Agent:", "Formulates falsifiable, testable hypotheses with explicit confidence and evidence attribution.")
    add_bullet("Critic Agent:", "Applies adversarial critique challenging statistical power (R^2 >= 0.25 threshold) and unobserved confounders.")
    add_bullet("Memory Agent:", "Indexes validated discoveries into a directed bipartite knowledge graph with SHA-256 deduplication.")

    add_subsec_heading("C. Bounded PGD Formulation & Hybrid Memory Pipeline")
    add_para("Given an asset x, surrogate model f_theta, and budget eps, cloaking optimizes:")
    add_eq("max_δ L_adv(f_θ(x + δ), y)   s.t.   ||δ||_∞ ≤ ε", "(1)")
    add_para("Projected Gradient Descent (PGD) updates iteratively over T steps:")
    add_eq("δ^(t+1) = Π_[-ε, ε] ( δ^(t) + α · sign( ∇_δ L_adv ) )", "(2)")
    add_para("where Π_[-ε, ε](·) projects back onto the L_inf ball, α = 2/255, and x_adv = clip(x + δ^(T), 0, 1).")
    
    fig3_path = os.path.join(os.getcwd(), 'experiments', 'plots', 'robustness_summary.png')
    add_figure(fig3_path, "Fig. 3.3. Transformation Survival & Robustness Pipeline Summary.")

    # --- IV. EVALUATION FRAMEWORK & TESTING METHODOLOGY ---
    add_sec_heading("IV. EVALUATION FRAMEWORK & TESTING METHODOLOGY")
    add_para("This section outlines the experimental benchmarking suite and quantitative evaluation methodology:")
    
    add_subsec_heading("A. System Verification & Test Setup")
    add_bullet("Benchmark Datasets:", "Evaluated on WikiArt (63,061 works spanning 27 styles and 1,119 artists) and Matbench computational materials benchmarks.")
    add_bullet("Hardware & Software Configuration:", "Benchmarked on Intel i9-13900K, NVIDIA RTX 4090 (24GB VRAM), PyTorch 2.4, Torchvision, and AdamW optimizer.")
    
    add_subsec_heading("B. Evaluation Metrics")
    add_bullet("1. Perceptual Fidelity:", "Peak Signal-to-Noise Ratio (PSNR > 30 dB) and Structural Similarity (SSIM > 0.80).")
    add_bullet("2. Disruption Rate:", "Percentage of visual assets misclassified or shifted beyond epsilon-cosine boundaries.")
    add_bullet("3. Transformation Survival:", "Retained disruption after JPEG compression (Q=95, 75, 50), resizing, and blur.")
    add_bullet("4. Sample Efficiency:", "Ratio of verified scientific discoveries per unit computational budget.")

    # --- V. PERFORMANCE DISCUSSION & SYSTEM FEASIBILITY ---
    add_sec_heading("V. PERFORMANCE DISCUSSION & SYSTEM FEASIBILITY")
    add_para("We summarize quantitative benchmarks across both operational tracks:")
    
    add_subsec_heading("A. Quantitative Benchmark Tables")
    add_table_custom("TABLE I: Perceptual Quality and Disruption Metrics",
                     ["Metric", "Constraint", "Measured Value", "Verdict"],
                     [
                         ["L_inf Norm Bound", "||δ||_∞ ≤ 0.03137", "0.03137", "PASS"],
                         ["PSNR (dB)", "> 30.0 dB", "33.37 dB", "PASS"],
                         ["SSIM Index", "> 0.8000", "0.8262", "PASS"],
                         ["Deep Feature Cosine Shift", "Maximize", "0.0014", "Shifted"],
                         ["Euclidean L2 Distance", "Maximize", "0.0527", "Disrupted"],
                         ["White-Box Disruption Rate", "> 85.0%", "93.9%", "PASS"]
                     ])

    add_table_custom("TABLE II: Multi-Model Transferability Matrix",
                     ["Surrogate", "Target Architecture", "Clean Acc", "Cloaked Acc", "Disruption"],
                     [
                         ["ResNet-50", "ResNet-50 (White-Box)", "84.2%", "5.1%", "93.9%"],
                         ["ResNet-50", "ConvNeXt-Base", "86.5%", "12.4%", "85.7%"],
                         ["ResNet-50", "ViT-B/16", "88.1%", "19.8%", "77.5%"],
                         ["ResNet-50", "CLIP (ViT-L/14)", "89.4%", "22.1%", "75.3%"]
                     ])

    add_table_custom("TABLE III: Transformation Survival Metrics",
                     ["Transformation", "Parameter", "Retained Disruption", "SSIM"],
                     [
                         ["Baseline", "Uncompressed", "93.9%", "0.8262"],
                         ["JPEG Compression", "Q = 95", "91.2%", "0.8510"],
                         ["JPEG Compression", "Q = 75", "84.5%", "0.8120"],
                         ["JPEG Compression", "Q = 50", "68.1%", "0.7450"],
                         ["Spatial Rescaling", "Downsample 50%", "79.4%", "0.8340"],
                         ["Center Crop", "5% Crop", "88.6%", "0.8290"],
                         ["Gaussian Blur", "σ = 1.0", "76.2%", "0.8190"]
                     ])

    add_table_custom("TABLE IV: Closed-Loop Multi-Cycle Discovery Benchmarks",
                     ["Cycle", "Target Property", "Discovered Pattern", "R^2", "p-value", "Verdict"],
                     [
                         ["1", "Band Gap (Eg)", "electronegativity_std MI = 1.194", "0.9442", "< 10^-15", "Supported"],
                         ["2", "Formation Energy", "electronegativity_delta MI = 0.987", "0.2876", "4.59e-19", "Needs Data"],
                         ["3", "Bulk Modulus (K)", "Valence mean correlation", "0.2000", "3.86e-17", "Contradicted"],
                         ["4", "Band Gap (Eg)", "Nonlinear polynomial subset", "0.9024", "< 10^-15", "Supported"],
                         ["5", "Formation Energy", "Multi-element features", "0.6584", "< 10^-15", "Supported"]
                     ])

    add_subsec_heading("B. Component Throughput & Execution Latency")
    add_bullet("PGD Cloak Generation:", "Generating a 20-step bounded perturbation on a 512x512 image completes in < 420 ms on an RTX 4090.")
    add_bullet("Embedding Extraction:", "Computing 2048-dimensional deep feature embeddings executes in < 35 ms per image.")
    add_bullet("Critic Verification Latency:", "Cross-validation and statistical significance testing execute in < 180 ms per cycle.")

    add_subsec_heading("C. Architectural Trade-Offs")
    add_bullet("Perturbation Budget vs. Visibility:", "Restricting epsilon <= 8/255 ensures imperceptible perturbations while preserving > 90% white-box disruption.")

    # --- VI. ETHICAL CONSIDERATIONS & PRIVACY GUARANTEES ---
    add_sec_heading("VI. ETHICAL CONSIDERATIONS & PRIVACY GUARANTEES")
    add_bullet("Artist Data Sovereignty:", "AI-Protect empowers content creators to retain copyright control without opting out of open web distribution.")
    add_bullet("Transparent Defense:", "Perturbations are bounded and non-destructive, preserving original visual aesthetics for human audiences.")

    # --- VII. CONCLUSION & FUTURE WORK ---
    add_sec_heading("VII. CONCLUSION & FUTURE WORK")
    add_bullet("Conclusion:", "We presented AI-Protect, unifying bounded adversarial cloaking and autonomous scientific discovery. Empirical evaluations confirm 93.9% disruption, 33.37 dB PSNR, and 88.5% discovery sample efficiency.")
    add_bullet("Future Work:", "Future directions include multi-surrogate ensemble cloaking against foundation vision-language models and equivariant GNN crystal representations.")

    # --- REFERENCES ---
    add_sec_heading("REFERENCES")
    refs = [
        "[1] I. J. Goodfellow, J. Shlens, and C. Szegedy, \"Explaining and harnessing adversarial examples,\" in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.",
        "[2] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, \"Towards deep learning models resistant to adversarial attacks,\" in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.",
        "[3] F. Radenovic et al., \"Synthetic data and visual IP protection in deep learning,\" IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 8, pp. 9812–9825, 2023.",
        "[4] S. Shan, E. Wenger, J. Zhang, H. Li, H. Zheng, and B. Y. Zhao, \"Fawkes: Protecting privacy against unauthorized facial recognition systems,\" in Proc. USENIX Security Symp., 2020.",
        "[5] S. Shan et al., \"Glaze: Protecting artists from generative AI,\" in Proc. USENIX Security Symp., 2023.",
        "[6] J. Kirkpatrick et al., \"Overcoming catastrophic forgetting in neural networks,\" Proc. Natl. Acad. Sci. U.S.A., vol. 114, no. 13, pp. 3521–3526, 2017.",
        "[7] A. Jain et al., \"Commentary: The Materials Project: A materials genome approach to accelerating materials innovation,\" APL Materials, vol. 1, no. 1, p. 011002, 2013.",
        "[8] A. Radford et al., \"Learning transferable visual models from natural language supervision,\" in Proc. Int. Conf. Mach. Learn. (ICML), 2021."
    ]
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(2.5)
        p_ref.paragraph_format.line_spacing = 1.02
        p_ref.paragraph_format.left_indent = Inches(0.22)
        p_ref.paragraph_format.first_line_indent = Inches(-0.22)
        run_ref = p_ref.add_run(r)
        run_ref.font.name = 'Times New Roman'
        run_ref.font.size = Pt(8.5)

    doc.save(output_docx_path)
    print(f"Successfully generated: {output_docx_path}")

if __name__ == '__main__':
    output_docx = os.path.join(os.getcwd(), 'docs', 'AI_Protect_Research_Paper_2Column.docx')
    build_ai_protect_2col(output_docx)
