# 🚀 Deployment Guide: Vaccine Pipeline Platform

## Free Deployment Options

### Option 1: Streamlit Cloud (Recommended - Easiest & Free)

**Streamlit Cloud** is the official free hosting platform for Streamlit apps. Perfect for your board meeting presentation!

#### Steps:

1. **Create a GitHub Account** (if you don't have one)
   - Go to https://github.com
   - Sign up for free

2. **Create a New Repository**
   - Click "New repository"
   - Name it: `vaccine-pipeline-platform`
   - Make it **Public** (required for free Streamlit Cloud)
   - Don't initialize with README (we'll add files manually)

3. **Upload Your Code to GitHub**
   
   **Option A: Using GitHub Desktop (Easiest)**
   - Download GitHub Desktop: https://desktop.github.com
   - Clone your repository
   - Copy all files from `C:\Users\amant\Desktop\demo` to the repository folder
   - Commit and push

   **Option B: Using Git Command Line**
   ```bash
   cd C:\Users\amant\Desktop\demo
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/vaccine-pipeline-platform.git
   git push -u origin main
   ```

4. **Create `.streamlit/secrets.toml` file** (IMPORTANT!)
   - In your repository, create a folder `.streamlit`
   - Create file `secrets.toml` inside it
   - Add your API key:
   ```toml
   OPENROUTER_API_KEY = "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"
   ```
   - **DO NOT** commit your `.env` file to GitHub (add it to `.gitignore`)

5. **Create `.gitignore` file** (in root of repository)
   ```
   .env
   .venv/
   __pycache__/
   *.pyc
   .streamlit/secrets.toml
   ```

6. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `vaccine-pipeline-platform`
   - Main file path: `test.py`
   - Click "Deploy"
   - Wait 2-3 minutes for deployment

7. **Add Secrets to Streamlit Cloud**
   - In Streamlit Cloud dashboard, go to your app settings
   - Click "Secrets" tab
   - Add:
   ```toml
   OPENROUTER_API_KEY = "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"
   ```
   - Save and restart the app

8. **Get Your Live URL**
   - Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`
   - Share this URL for your board meeting!

#### Advantages:
- ✅ Completely free
- ✅ Automatic HTTPS
- ✅ Auto-deploys on git push
- ✅ No server management
- ✅ Fast and reliable
- ✅ Custom domain support (optional)

---

### Option 2: Hugging Face Spaces (Alternative Free Option)

1. Go to https://huggingface.co/spaces
2. Create new Space
3. Select "Streamlit" SDK
4. Upload your files
5. Add secrets in Settings

---

### Option 3: Railway.app (Free Tier Available)

1. Sign up at https://railway.app
2. New Project → Deploy from GitHub
3. Select your repository
4. Add environment variables
5. Deploy

---

## Pre-Deployment Checklist

- [ ] All code is in GitHub repository
- [ ] `.gitignore` includes `.env` and `.venv/`
- [ ] `requirements.txt` is up to date
- [ ] API key is added to Streamlit Cloud secrets (NOT in code)
- [ ] Test locally: `streamlit run test.py`
- [ ] Verify all features work

## Security Best Practices

1. **Never commit API keys to GitHub**
   - Use `.gitignore` to exclude `.env`
   - Use Streamlit Cloud Secrets for production

2. **Keep secrets secure**
   - Rotate API keys periodically
   - Use different keys for dev/prod if needed

3. **Monitor usage**
   - Check OpenRouter dashboard for API usage
   - Set up alerts if needed

## Troubleshooting

### App won't deploy
- Check `requirements.txt` syntax
- Verify main file path is correct (`test.py`)
- Check Streamlit Cloud logs

### API key not working
- Verify secret name matches: `OPENROUTER_API_KEY`
- Check secret is saved in Streamlit Cloud
- Restart the app after adding secrets

### Import errors
- Ensure all packages in `requirements.txt`
- Check Python version compatibility

## Quick Start Commands

```bash
# Test locally
streamlit run test.py

# Check requirements
pip install -r requirements.txt

# Verify .env is loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key:', os.getenv('OPENROUTER_API_KEY')[:20] + '...')"
```

## Support

For Streamlit Cloud issues: https://discuss.streamlit.io
For deployment help: Check Streamlit Cloud documentation

---

**Recommended: Use Streamlit Cloud for easiest deployment! 🚀**

