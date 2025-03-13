<p align="center">
  <img align="center" src="https://docs.keploy.io/img/keploy-logo-dark.svg?s=200&v=4" height="40%" width="40%"  alt="keploy logo"/>  <!-- we can add banner here, maybe a poster or a gif -->
</p>
<h3 align="center">
<b>
⚡️ KEPLOY RAG BOT ⚡️
</b>
</h3 >
<p align="center">
🌟 Rag bot agent for <a href="https://keploy.io">Keploy.io</a> website that answers all the queries on the landing page itself🌟
</p>

---

## About Keploy RAGBot
Keploy RAGBot is an AI-powered chatbot built using Python to handle user queries related to Keploy. Integrated into the official Keploy website landing page, this bot provides instant and accurate responses to user questions, enhancing the user experience and engagement.

## Features
- **Retrieval-Augmented Generation (RAG)** for improved answer accuracy.
- **Seamless integration** with Keploy's official website.
- **Fast and efficient** responses to user queries about Keploy.
- **Scalable and customizable** to accommodate evolving user needs.

## Getting Started
Follow these steps to set up and run the Keploy RAGBot on your local machine.

### Prerequisites
Ensure you have the following installed before proceeding:
- Python 3.8+
- pip (Python package manager)
- Git

### Installation & Setup
#### 1. Clone the Repository
```bash
 git clone https://github.com/keploy/keploy-ragbot.git
 cd keploy-ragbot
```
#### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
```
#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
#### 4. Run the Bot Locally
```bash
python app.py
```
The bot should now be running locally on the `localhost `.

## Folder Structure
```
keploy-ragbot/
│-- .github/workflows/     # GitHub Actions for CI/CD
│-- docs/                  # Documentation files
│-- document_index/        # Index files for document retrieval
│-- app.py                 # Main application file
│-- brain.py               # Core logic of the chatbot
│-- index.faiss            # FAISS index for fast search
│-- index.pkl              # Serialized index for RAG
│-- requirements.txt       # Dependencies
│-- vercel.json            # Deployment configuration for Vercel
│-- Dockerfile             # Docker container setup
│-- .gitignore             # Ignored files
```

## Contributing
We welcome contributions! Follow these steps to contribute:

1. **Fork the Repository** on GitHub.
2. **Clone Your Fork** to your local system.
3. **Create a New Branch** for your changes.
   ```bash
   git checkout -b feature-name
   ```
4. **Make Your Changes** and commit them.
5. **Push Your Changes** to your fork.
   ```bash
   git push origin feature-name
   ```
6. **Create a Pull Request** to the main repository.


---

For any issues or feature requests, feel free to open an issue in the repository!

You can also reach out to us for discussions on our <a href="https://join.slack.com/t/keploy/shared_invite/zt-2poflru6f-_VAuvQfCBT8fDWv1WwSbkw" alt="Slack">
<img src="https://img.shields.io/badge/Slack-@layer5.svg?logo=slack" />
</a>
