# 🚀 Streamlit Cloud Deployment - Step by Step

## Using the "Deploy" Button in Streamlit App (Easiest Method!)

### Step 1: Prepare Your Code
1. Make sure all your files are saved
2. Ensure `.env` file exists with your API key (for local testing)
3. Verify `requirements.txt` is up to date

### Step 2: Create GitHub Repository

1. **Go to GitHub.com** and sign in (or create account)
2. **Click the "+" icon** in top right → "New repository"
3. **Repository name**: `vaccine-pipeline-platform`
4. **Make it Public** (required for free Streamlit Cloud)
5. **DO NOT** check "Initialize with README" (we'll add files manually)
6. **Click "Create repository"**

### Step 3: Upload Your Code to GitHub

**Option A: Using GitHub Desktop (Recommended for beginners)**

1. Download GitHub Desktop: https://desktop.github.com
2. Install and sign in with your GitHub account
3. Click "File" → "Clone repository"
4. Select your new repository
5. Choose a local folder (e.g., `C:\Users\amant\Desktop\github\vaccine-pipeline-platform`)
6. Copy ALL files from `C:\Users\amant\Desktop\demo` to the cloned folder:
   - `test.py`
   - `requirements.txt`
   - `.gitignore`
   - `README.md`
   - `DEPLOYMENT.md`
   - **DO NOT copy**: `.env`, `.venv/` (these are in .gitignore)
7. In GitHub Desktop:
   - Click "Commit to main"
   - Write commit message: "Initial commit"
   - Click "Push origin"

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

### Step 4: Deploy to Streamlit Cloud

1. **Go to your Streamlit app** (the one running locally)
2. **Click the hamburger menu (☰)** in top right corner
3. **Click "Deploy this app"** or "Settings" → "Deploy app"
4. **Sign in to Streamlit Cloud** (use GitHub account)
5. **Authorize Streamlit** to access your GitHub repositories
6. **Select your repository**: `vaccine-pipeline-platform`
7. **Main file path**: `test.py`
8. **Click "Deploy"**

### Step 5: Add Your API Key (CRITICAL!)

1. **In Streamlit Cloud dashboard**, go to your app
2. **Click the "⋮" (three dots)** next to your app name
3. **Click "Settings"**
4. **Click "Secrets" tab**
5. **Add this** (replace with your actual key):
   ```toml
   OPENROUTER_API_KEY = "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"
   ```
6. **Click "Save"**
7. **Wait for app to restart** (automatic)

### Step 6: Get Your Live URL

1. **Your app is now live!**
2. **Copy the URL** from Streamlit Cloud dashboard
3. **Format**: `https://YOUR_APP_NAME.streamlit.app`
4. **Share this URL** for your board meeting!

## Troubleshooting

### "Deploy" button not showing?
- Make sure you're signed in to Streamlit Cloud
- Try refreshing the page
- Use the manual method below

### Manual Deployment (If button doesn't work)

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `vaccine-pipeline-platform`
5. Main file: `test.py`
6. Click "Deploy"

### App not working after deployment?

1. **Check logs**: Click "Manage app" → "Logs"
2. **Verify secrets**: Settings → Secrets (make sure API key is there)
3. **Check requirements.txt**: Make sure all packages are listed
4. **Restart app**: Settings → "Restart app"

### API Key Issues?

- Make sure it's in Streamlit Cloud Secrets (not in code)
- Format: `OPENROUTER_API_KEY = "your-key-here"`
- No quotes around the key value
- Restart app after adding secrets

## Quick Checklist

- [ ] GitHub repository created and public
- [ ] All files uploaded to GitHub (except .env)
- [ ] Streamlit Cloud app deployed
- [ ] API key added to Secrets
- [ ] App restarted after adding secrets
- [ ] Test the live URL
- [ ] PDF downloads work
- [ ] Summaries generate correctly

## Your Live App URL

Once deployed, your app will be at:
```
https://YOUR_APP_NAME.streamlit.app
```

**Bookmark this URL for your board meeting!** 🎯

---

**Need help?** Check the main DEPLOYMENT.md file for more details.

