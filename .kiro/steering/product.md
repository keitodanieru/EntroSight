# EntroSight — Product Summary

EntroSight is a privacy-preserving Windows PE malware family classifier with explainable threat intelligence.

## What It Does

- Accepts uploaded PE binaries (.exe, .dll, .sys)
- Converts them to byte-entropy heatmaps
- Classifies into malware families (or benign) using a fine-tuned ResNet50 model
- Generates plain-language threat explanations using RAG + Ollama LLM
- Stores scan history for review and deduplication

## Key Principles

- **Privacy-preserving**: Uploaded binaries are deleted immediately after feature extraction. No data leaves the local environment.
- **Explainable AI**: Classifications are accompanied by MITRE ATT&CK-backed explanations, not just labels.
- **Local-only**: All inference, retrieval, and generation run on-premises via Docker Compose.

## Supported Malware Families

AgentTesla, Remcos, DCRat, AsyncRAT, RedLineStealer, Formbook, and Benign.

## Target Performance

- Classification: under 1 second (CPU)
- Full scan-to-explanation pipeline: under 30 seconds
