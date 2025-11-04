# MedDialogue Documentation Website

This directory contains the GitHub Pages documentation website for MedDialogue.

## Files

- `index.html` - Landing page with features and overview
- `quick-start.html` - Quick start guide for new users
- `technical.html` - Technical documentation
- `api.html` - API reference
- `css/style.css` - Styling for all pages
- `_config.yml` - GitHub Pages configuration

## Deployment

### Automatic Deployment (GitHub Pages)

1. Push this directory to your repository
2. Go to repository Settings → Pages
3. Set source to: **Deploy from a branch**
4. Select branch: `main` (or your branch)
5. Select folder: `/docs`
6. Click Save

Your site will be available at: `https://<username>.github.io/<repository>/`

### Local Testing

To test locally before deploying:

```bash
# Install jekyll (if not already installed)
gem install jekyll bundler

# Serve the site locally
cd docs
jekyll serve

# Visit http://localhost:4000
```

## Updating Documentation

1. Edit the HTML files directly
2. Test changes locally (optional)
3. Commit and push to trigger automatic deployment

## Customization

### Colors
Edit `css/style.css` and modify the `:root` variables:
```css
:root {
    --primary-color: #2563eb;
    --secondary-color: #10b981;
    ...
}
```

### Content
Edit the HTML files directly. All pages share the same navigation and footer structure.

## Support

For issues or questions:
- GitHub Issues: https://github.com/musc-bmic/meddialogue/issues
- Email: gyasi@musc.edu
