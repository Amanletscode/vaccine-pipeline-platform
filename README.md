# 💉 Vaccine Pipeline Platform

A comprehensive intelligence platform for vaccine clinical trial analysis, competitive intelligence, and executive reporting.

## Features

- 🔍 **Disease & Vaccine Search**: Search clinical trials by disease condition or vaccine product name
- 🧠 **AI-Powered Summaries**: Executive-ready summaries with competitive intelligence insights
- 📊 **Interactive Visualizations**: Phase distribution, status charts, and sponsor analytics
- 📄 **Professional PDF Reports**: Download beautifully formatted PDF reports for presentations
- 🔬 **Detailed Trial Analysis**: Comprehensive trial details including enrollment, design, eligibility, and outcomes
- 🆚 **Competitor Analysis**: Automatic competitor identification and comparison



### Local Setup

1. **Clone or download this repository**

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a `.env` file in the root directory
   - Add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

5. **Run the app**
   ```bash
   streamlit run test.py
   ```

6. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

## Configuration

### API Keys

The app uses OpenRouter for AI summarization. Your API key should be stored in:

- **Local development**: `.env` file (not committed to git)
- **Production/Deployment**: Streamlit Cloud Secrets (see DEPLOYMENT.md)

### Environment Variables

Create a `.env` file with:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

## Usage

### Search by Disease
1. Enter a disease name (e.g., "RSV", "COVID-19")
2. Click "Fetch All Trials"
3. Filter by phase and status
4. View visualizations and generate AI summaries
5. Download PDF reports

### Search by Vaccine Product
1. Enter a vaccine product name
2. Click "Search Vaccine & Competitors"
3. View your vaccine's trials and competitor analysis
4. Generate comprehensive summaries
5. Export PDF reports

## Deployment

For free deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

**Recommended**: Streamlit Cloud (completely free, easy setup)

## Technology Stack

- **Frontend**: Streamlit
- **Data Source**: ClinicalTrials.gov API v2
- **AI/LLM**: OpenRouter API
- **Visualization**: Plotly
- **PDF Generation**: ReportLab
- **Data Processing**: Pandas, NumPy

## Roadmap

See the "Platform Roadmap" section in the app for detailed RAG application development plan.

## Security Notes

- ⚠️ Never commit `.env` files or API keys to version control
- ✅ Use `.gitignore` to exclude sensitive files
- ✅ Use Streamlit Cloud Secrets for production deployments

## Support

For issues or questions:
- Check the deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- Review Streamlit documentation: https://docs.streamlit.io

## License

Developed by Aman




