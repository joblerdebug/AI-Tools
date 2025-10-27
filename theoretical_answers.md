# Part 1: Theoretical Understanding

## Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

**TensorFlow:**
- Uses static computation graphs (define-then-run)
- Better for production deployment and mobile applications
- Strong ecosystem with TensorFlow Serving, TF Lite, TF.js
- More verbose syntax, steeper learning curve

**PyTorch:**
- Uses dynamic computation graphs (define-by-run)
- More intuitive, Pythonic, and flexible for research
- Easier debugging and prototyping
- Preferred for academic research and rapid experimentation

**When to choose:**
- Choose **PyTorch** for research, prototyping, and when you need flexibility
- Choose **TensorFlow** for production systems, mobile deployment, and enterprise applications

## Q2: Describe two use cases for Jupyter Notebooks in AI development.

1. **Exploratory Data Analysis (EDA):**
   - Interactively explore datasets with immediate visual feedback
   - Create data visualizations, check for missing values, understand distributions
   - Combine code, charts, and markdown explanations in one document

2. **Model Prototyping and Tutorials:**
   - Step-by-step model development with immediate execution
   - Share reproducible research with code, results, and explanations
   - Create interactive demonstrations and educational materials

## Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

**Basic String Operations:**
- Simple pattern matching (e.g., `string.find()`, `string.split()`)
- No understanding of language structure or context
- Limited to exact matches and regular expressions

**spaCy Enhancements:**
- **Linguistic Understanding:** Part-of-speech tagging, dependency parsing
- **Named Entity Recognition:** Identifies people, organizations, products, locations
- **Context Awareness:** Understands word relationships and sentence structure
- **Pre-trained Models:** Accurate out-of-the-box performance on various tasks
- **Efficiency:** Optimized for large-scale text processing

## Comparative Analysis: Scikit-learn vs TensorFlow

| Aspect | Scikit-learn | TensorFlow |
|--------|-------------|------------|
| **Target Applications** | Classical ML algorithms (SVM, decision trees, linear regression) | Deep learning (CNNs, RNNs, transformers) |
| **Ease of Use** | Simple, consistent API ideal for beginners | Steeper learning curve, but Keras provides high-level interface |
| **Community Support** | Excellent documentation for traditional ML | Larger community for cutting-edge deep learning research |
| **Performance** | Optimized for small to medium datasets on CPU | Designed for large-scale training on GPUs/TPUs |
| **Deployment** | Simple model serialization with pickle | Production-ready with TensorFlow Serving, TF Lite |
