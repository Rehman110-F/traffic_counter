// Replace with your actual API URL
const apiUrl = "http://127.0.0.1:8000/counts";

// Store current data to prevent flickering
let currentData = {};
let isFirstLoad = true;

async function loadData() {
  const container = document.getElementById("data-container");
  const updatedText = document.getElementById("last-updated");
  
  // Only show loading spinner on first load
  if (isFirstLoad) {
    container.innerHTML = `
      <div class="loading-spinner">
        <div class="spinner"></div>
        <span>Loading data...</span>
      </div>
    `;
  }

  try {
    const response = await fetch(apiUrl);
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    const data = await response.json();

    // Check if data has actually changed
    const dataChanged = JSON.stringify(data) !== JSON.stringify(currentData);
    
    if (dataChanged || isFirstLoad) {
      currentData = data;
      updateDisplay(data);
    }

    // Always update the timestamp
    const now = new Date().toLocaleTimeString();
    updatedText.innerHTML = `🕒 Last updated: ${now}`;
    
    isFirstLoad = false;
  } catch (error) {
    container.innerHTML = `<div class="error">Connection Error<br><small>${error.message}</small></div>`;
    updatedText.innerHTML = "";
  }
}

function updateDisplay(data) {
  const container = document.getElementById("data-container");
  
  // Clear container
  container.innerHTML = "";
  
  // Check if data is empty
  if (Object.keys(data).length === 0) {
    container.innerHTML = `
      <div class="data-item">
        <div class="label">No Data Available</div>
        <div class="count">—</div>
      </div>
    `;
    return;
  }

  // Create data items with animation
  Object.keys(data).forEach((key, index) => {
    const div = document.createElement("div");
    div.className = "data-item";
    
    // Add different vehicle icons based on the key name
    let icon = "🚗";
    const keyLower = key.toLowerCase();
    if (keyLower.includes("car")) icon = "🚗";
    else if (keyLower.includes("truck")) icon = "🚛";
    else if (keyLower.includes("bus")) icon = "🚌";
    else if (keyLower.includes("bike") || keyLower.includes("motorcycle")) icon = "🏍️";
    else if (keyLower.includes("person") || keyLower.includes("people")) icon = "🚶";
    else if (keyLower.includes("total")) icon = "📊";
    
    div.innerHTML = `
      <div class="label">
        <span style="margin-right: 0.5rem;">${icon}</span>
        ${formatLabel(key)}
      </div>
      <div class="count" data-count="${data[key]}">${data[key]}</div>
    `;
    
    // Add staggered animation
    div.style.animationDelay = `${index * 0.1}s`;
    div.style.animation = 'slideUp 0.6s ease-out forwards';
    
    container.appendChild(div);
  });
  
  // Animate count changes
  animateCounters();
}

function formatLabel(key) {
  // Convert snake_case or camelCase to Title Case
  return key
    .replace(/([A-Z])/g, ' $1') // Add space before capital letters
    .replace(/[_-]/g, ' ') // Replace underscores and hyphens with spaces
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
    .trim();
}

function animateCounters() {
  const counters = document.querySelectorAll('.count');
  
  counters.forEach(counter => {
    const target = parseInt(counter.dataset.count) || 0;
    const current = parseInt(counter.textContent) || 0;
    
    if (target !== current) {
      animateCounter(counter, current, target, 300); // 300ms animation
    }
  });
}

function animateCounter(element, start, end, duration) {
  const range = end - start;
  const startTime = performance.now();
  
  function updateCounter(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // Easing function for smooth animation
    const easeOut = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (range * easeOut));
    
    element.textContent = current;
    
    if (progress < 1) {
      requestAnimationFrame(updateCounter);
    } else {
      element.textContent = end; // Ensure final value is exact
    }
  }
  
  requestAnimationFrame(updateCounter);
}

// Add refresh animation
function animateRefreshIcon() {
  const refreshIcon = document.querySelector('.refresh-icon');
  if (refreshIcon) {
    refreshIcon.style.animation = 'none';
    setTimeout(() => {
      refreshIcon.style.animation = 'rotate 1s linear';
    }, 10);
  }
}

// Call immediately on page load
loadData();

// Refresh every 5 seconds with visual feedback
setInterval(() => {
  animateRefreshIcon();
  loadData();
}, 5000);

// Add some interactive effects
document.addEventListener('DOMContentLoaded', function() {
  // Add click effect to data items
  document.addEventListener('click', function(e) {
    if (e.target.closest('.data-item')) {
      const item = e.target.closest('.data-item');
      item.style.transform = 'scale(0.98)';
      setTimeout(() => {
        item.style.transform = '';
      }, 150);
    }
  });
  
  // Add keyboard accessibility
  document.addEventListener('keydown', function(e) {
    if (e.key === 'r' || e.key === 'R') {
      if (e.ctrlKey || e.metaKey) return; // Don't interfere with browser refresh
      loadData();
      animateRefreshIcon();
    }
  });
});