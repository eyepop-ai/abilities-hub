# EyePop Example Abilities

Production-ready visual intelligence building blocks for real-world applications.

This repository contains curated examples of **EyePop.ai Abilities** designed to help developers move from raw image or video input to structured, reliable outputs — quickly.

Half inspiration. Half implementation guide.

If you're exploring what’s possible with Vision Language Models (VLMs) and computer vision, this repo shows you.  
If you're building production systems, this repo shows you how.

---

## 🚀 What This Repo Is

This is a collection of:

- Image classification abilities  
- Object detection abilities  
- Temporal video event detection  
- Action recognition  
- Document and ID structured data extraction  
- Prompt-driven video segmentation  
- Evaluation-ready structured outputs  

Each example demonstrates a **single, clearly defined capability** implemented with:

- Explicit input assumptions  
- Deterministic output formats  
- Structured JSON responses  
- Negative case handling  
- Production-grade prompt design  

These are not toy demos.  
They are composable primitives you can deploy into real systems.

---

## 🎯 Why This Matters

Vision systems fail at ambiguity.

Most computer vision projects break not because the model is incapable — but because:

- The prompt is underspecified  
- Edge cases are undefined  
- Outputs are unconstrained  
- Evaluation criteria are unclear  

This repository exists to close that gap.

Every example here treats:

- Prompts as software  
- Outputs as contracts  
- Edge cases as first-class citizens  
- Evaluation as mandatory  

---

## 🧠 Design Philosophy

We follow five principles:

### 1. Constrain Outputs
Models are probabilistic. Production systems cannot be.

Every ability defines:
- Exactly what labels are allowed  
- When to return `NO`  
- When to return `None`  
- When to abstain  

### 2. Define the Negative Case
Ambiguity causes drift.  
Each ability explicitly defines when something is *not* happening.

### 3. Prefer Structured Data Over Prose
Structured JSON > paragraphs.

Abilities return:
- Labeled classifications  
- Confidence scores  
- Structured extracted fields  
- Fixed schemas  

### 4. Optimize for Reproducibility
Every example is:
- Evaluation-ready  
- Compatible with SDK workflows  
- Suitable for cloud or on-prem runtime  

### 5. Treat Prompt Engineering as Engineering
Prompts are versioned.  
Schemas are contracts.  
Failure modes are documented.  

---

## 🏗 Repository Structure

Each ability example includes:

- `README.md` explaining the use case  
- The production prompt  
- Example input  
- Example output  
- Expected JSON schema  
- Notes on evaluation  

Where applicable, examples demonstrate:

- One-second video window classification  
- Region-of-interest cropping strategies  
- Label precedence rules  
- Confidence thresholds  
- Deterministic return behavior  

---

## 💡 Example Use Cases

You’ll find patterns applicable to:

- Sports broadcast segmentation  
- Industrial event counting  
- Retail checkout analysis  
- Medical action detection  
- Structured document extraction  
- ID and vehicle title parsing  
- Video Q&A without model training  

Each example is intentionally small. The goal is composability.

You can combine multiple abilities to build:

- Automated review pipelines  
- Real-time monitoring systems  
- Analytics dashboards  
- Event detection engines  
- Compliance workflows  

---

## 🛠 Who This Is For

This repository is built for:

### Developers
- Integrating EyePop into production systems  
- Designing structured VLM prompts  
- Building evaluation loops  
- Deploying to edge or cloud environments  

### Product Teams
- Prototyping vertical use cases  
- Converting business intent into model instructions  
- Reducing hallucination risk  

### Enterprise Teams
- Designing repeatable visual intelligence workflows  
- Implementing governance and traceability  
- Creating audit-friendly outputs  

---

## ⚙ Deployment Compatibility

All examples are designed to work with:

- EyePop VLM Cloud  
- On-prem AI Application Runtime  
- SDK-based integrations  
- Evaluation pipelines  
- Dataset-driven iteration  

They are intentionally written to be:

- Copy-paste deployable  
- Versionable  
- Measurable  
- Safe to integrate into production systems  

---

## 📊 How to Use This Repo

1. Identify a use case similar to yours  
2. Review the ability structure  
3. Adapt the prompt and schema to your domain  
4. Add ground truth examples  
5. Run evaluation  
6. Iterate deterministically  

This is not about prompting once.  
It is about designing a system.

---

## 🔬 From Example to Production

If you want to move beyond examples:

- Add dataset-backed evaluation  
- Introduce confidence gating  
- Define escalation paths for `NO` or ambiguous cases  
- Log trace IDs  
- Version prompts alongside code  

Visual intelligence becomes reliable when treated as infrastructure, not novelty.

---

## 🌊 The Bigger Picture

We believe:

- You should not need to train a custom model to analyze video  
- Prompt-driven visual intelligence should be testable  
- Edge deployment should not require ML expertise  
- Structured outputs should be the default  

This repository demonstrates that philosophy in executable form.

---

## 📌 Final Note

These examples are intentionally opinionated.

They favor:
- Precision over cleverness  
- Determinism over creativity  
- Contracts over vibes  

Because production systems require discipline.

If you are building with visual AI and want it to behave like software — this is your starting point.
