# Graph Bundling Visualizer

A web-based interface for visualizing collections of graph bundling images with support for small multiples view and slider navigation.

## Features

### 🔄 Dual View Modes
- **Small Multiples View**: Display up to 9 images in a responsive grid layout
- **Slider View**: Navigate through images one at a time with thumbnail strip

### 🎛️ Interactive Controls
- **File Selection**: Load entire directories of images at once
- **Filtering**: Search and filter images by filename
- **Sorting**: Sort by name, date modified, or file size
- **Keyboard Navigation**: Use arrow keys, Home/End for navigation

### 🖼️ Image Features
- **Full-screen Modal**: Click any image to view in full resolution
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Image Information**: Display dimensions, file size, and metadata
- **Thumbnail Navigation**: Quick access to any image via thumbnail strip

### 🎨 User Experience
- **Modern UI**: Clean, professional interface with Bootstrap styling
- **Loading States**: Visual feedback during image processing
- **Toast Notifications**: Success/error messages for user actions
- **Accessibility**: Keyboard navigation and screen reader support

## Usage

1. **Open the visualizer** by opening `index.html` in a web browser
2. **Select images** using the "Image Directory" file input
3. **Choose view mode** using the Grid/Slider toggle buttons
4. **Filter and sort** images using the control panel
5. **Navigate** through images using buttons, slider, or keyboard

### Keyboard Shortcuts (Slider View)
- `←/→`: Previous/Next image
- `Home/End`: First/Last image
- `Escape`: Close modal

## File Structure
```
visualiser/
├── index.html      # Main HTML structure
├── styles.css      # CSS styling and responsive design
├── script.js       # JavaScript functionality
└── README.md       # This documentation
```

## Browser Compatibility
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Technical Features
- Pure JavaScript (no external dependencies except Bootstrap)
- Client-side image processing
- Responsive grid system
- File API for directory selection
- CSS Grid and Flexbox layouts
- Modern ES6+ JavaScript

## Integration with Graph Bundling Pipeline
This visualizer is designed to work with the output images from various bundling algorithms in the repository:
- EPB (Edge Path Bundling)
- Force-Directed Bundling
- WR (Winding Roads)
- CUBU bundling results

Simply select the output directory containing your bundling results to visualize and compare different algorithms and parameters.