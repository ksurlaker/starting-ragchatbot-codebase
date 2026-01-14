# Frontend Changes: Dark/Light Theme Toggle

## Overview
Added a theme toggle button that allows users to switch between dark and light themes with smooth transitions and persistent preference storage.

## Files Modified

### 1. `frontend/index.html`
- Added theme toggle button with sun/moon SVG icons positioned in the top-right corner
- Button includes proper accessibility attributes (`aria-label`, `title`)
- Updated CSS and JS version numbers to v10 for cache busting

### 2. `frontend/style.css`

#### New CSS Variables
Added theme-aware CSS variables in `:root` (dark theme) and `[data-theme="light"]` (light theme):
- `--code-bg`: Background color for code blocks
- `--source-item-bg` / `--source-item-hover`: Source item background colors
- `--source-link-color` / `--source-link-hover`: Source link colors

#### Light Theme Color Palette
```css
[data-theme="light"] {
    --background: #f8fafc;
    --surface: #ffffff;
    --surface-hover: #f1f5f9;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --assistant-message: #f1f5f9;
    --welcome-bg: #eff6ff;
    /* ... and more */
}
```

#### Theme Toggle Button Styling
- Fixed position in top-right corner (44x44px, circular)
- Smooth hover effects with scale transform
- Icon rotation animations when switching themes
- Focus states for keyboard accessibility
- Responsive sizing for mobile (40x40px)

#### Transition Animations
Added smooth 0.3s transitions to:
- Body background and text colors
- Sidebar, chat container, message content
- Input fields, buttons, stat items
- Source items and other interactive elements

### 3. `frontend/script.js`

#### New Functions
- `initializeTheme()`: Loads saved theme preference from localStorage on page load
- `toggleTheme()`: Switches between dark and light themes
- `setTheme(theme)`: Applies theme by setting/removing `data-theme` attribute on `<html>` element

#### Theme Persistence
- Theme preference saved to `localStorage` under key `'theme'`
- Default theme is dark if no preference is saved
- Preference persists across browser sessions

## Features

1. **Toggle Button Design**
   - Circular button with sun (light mode) / moon (dark mode) icons
   - Positioned in top-right corner, always visible
   - Smooth icon rotation animation on toggle

2. **Accessibility**
   - Keyboard navigable (Tab + Enter)
   - ARIA label for screen readers
   - Focus ring indicator
   - Proper color contrast in both themes

3. **Smooth Transitions**
   - 0.3s ease transitions on all color changes
   - No jarring flashes when switching themes
   - Icon swap animation with rotation effect

4. **Persistence**
   - Theme choice saved to localStorage
   - Automatically applied on page reload
   - Works across browser sessions

## Usage
Click the sun/moon button in the top-right corner to toggle between dark and light themes. Your preference will be remembered for future visits.
