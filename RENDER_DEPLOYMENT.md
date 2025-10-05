# Deploying Satellite Segmentation App to Render

## Prerequisites
- GitHub account
- Render account (sign up at https://render.com)
- Your GROQ API key from https://console.groq.com/keys

## Step 1: Prepare Your Repository

1. Make sure all files are committed to your GitHub repository
2. Verify `.env` file is in `.gitignore` (it should NOT be pushed to GitHub)

## Step 2: Create a Web Service on Render

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account if you haven't already
4. Select your repository: `nasafinalproject`

## Step 3: Configure Your Service

### Basic Settings:
- **Name**: `satellite-segmentation` (or your preferred name)
- **Region**: Choose closest to your users
- **Branch**: `main`
- **Root Directory**: Leave empty
- **Environment**: `Python 3`
- **Build Command**: `chmod +x build.sh && ./build.sh`
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

### Advanced Settings:

#### Instance Type
⚠️ **IMPORTANT**: Select **Starter ($7/month)** or higher
- Free tier (512 MB RAM) is insufficient for TensorFlow models
- Starter provides 2 GB RAM and no automatic sleep

#### Auto-Deploy
- Enable "Auto-Deploy" if you want automatic deployments on git push

## Step 4: Environment Variables

Click **"Advanced"** → **"Add Environment Variable"** and add these:

### Required Variables:

1. **GROQ_API_KEY**
   - Key: `GROQ_API_KEY`
   - Value: Your actual API key from GROQ console
   - Get it from: https://console.groq.com/keys

2. **SECRET_KEY**
   - Key: `SECRET_KEY`
   - Value: Generate a secure random string:
   ```python
   import secrets
   print(secrets.token_hex(32))
   ```

3. **FLASK_ENV**
   - Key: `FLASK_ENV`
   - Value: `production`

### Optional Variables:

4. **WORKERS** (if you want to customize)
   - Key: `WORKERS`
   - Value: `2` (adjust based on your instance size)

## Step 5: Deploy

1. Click **"Create Web Service"**
2. Render will start building your application
3. Monitor the logs in real-time
4. Deployment typically takes 5-10 minutes

## Step 6: Verify Deployment

Once deployment is complete:

1. Click on your service URL (e.g., `https://satellite-segmentation.onrender.com`)
2. Test the homepage loads
3. Try uploading a satellite image
4. Verify segmentation and AI analysis work

## Troubleshooting

### Build Fails

**Error: Module not found**
- Check `requirements.txt` has all dependencies
- Verify Python version compatibility

**Error: Out of memory**
- Upgrade to Starter plan (Free tier insufficient)
- Reduce number of workers in Procfile

### Application Crashes

**Error: Model file not found**
- Ensure `models/satellite_segmentation_full.h5` is in repository
- Check file size (GitHub has 100 MB limit per file)
- Consider using Git LFS for large model files

**Error: GROQ_API_KEY not found**
- Verify environment variable is set correctly in Render dashboard
- Check variable name matches exactly (case-sensitive)

### Performance Issues

**Slow response times**
- Increase worker count in Procfile
- Upgrade to higher-tier instance
- Consider adding caching

**Timeout errors**
- Increase timeout in Procfile: `--timeout 180`
- Check model loading time

## Monitoring

### Logs
- View real-time logs in Render dashboard
- Logs → View Logs

### Metrics
- Monitor CPU and memory usage
- Metrics tab shows performance graphs

### Alerts
- Set up email notifications for:
  - Deploy failures
  - Service crashes
  - High resource usage

## Updating Your Application

### Automatic Deployments
If auto-deploy is enabled:
```powershell
git add .
git commit -m "Update application"
git push origin main
```
Render will automatically redeploy.

### Manual Deployments
1. Go to Render dashboard
2. Select your service
3. Click "Manual Deploy" → "Deploy latest commit"

## Best Practices

### Security
- Never commit API keys to repository
- Use strong SECRET_KEY
- Keep dependencies updated
- Enable HTTPS (Render provides free SSL)

### Performance
- Use appropriate instance size for your traffic
- Monitor logs regularly
- Optimize model loading (consider caching)
- Use CDN for static assets if needed

### Cost Optimization
- Start with Starter plan ($7/month)
- Monitor usage and scale as needed
- Consider sleeping services if low traffic (Free tier only)

## Support Resources

- Render Documentation: https://render.com/docs
- Render Status: https://status.render.com
- Community Forum: https://community.render.com
- GROQ API Documentation: https://console.groq.com/docs

## Security Notes

⚠️ **NEVER** commit these files with real credentials:
- `.env` file
- Any file containing API keys
- Database credentials

Always use environment variables for sensitive data!

## Cost Estimate

- **Free Tier**: $0/month (limited resources, not recommended)
- **Starter**: $7/month (recommended for production)
- **Standard**: $25/month (for higher traffic)

Choose based on your expected traffic and resource needs.
