// Global variables
let map;
let drawnItems;
let currentBounds = null;

// Layer groups for each category
let layers = {
    healthcare: L.layerGroup(),
    food: L.layerGroup(),
    housing: L.layerGroup(),
    transportation: L.layerGroup(),
    parks: L.layerGroup(),
    industrial: L.layerGroup(),
    water: L.layerGroup(),
    agriculture: L.layerGroup(),
    waste: L.layerGroup(),
    energy: L.layerGroup()
};

// Suggestion layer groups (RED DOTS)
let suggestionLayers = {
    healthcare: L.layerGroup(),
    food: L.layerGroup(),
    parks: L.layerGroup(),
    transportation: L.layerGroup(),
    waste: L.layerGroup(),
    energy: L.layerGroup(),
    housing: L.layerGroup()
};

// Layer colors
const layerColors = {
    healthcare: '#007bff',
    food: '#28a745',
    housing: '#8b4513',
    transportation: '#6f42c1',
    parks: '#90ee90',
    industrial: '#ffc107',
    water: '#17a2b8',
    agriculture: '#d4a76a',
    waste: '#6c757d',
    energy: '#ff8c00'
};

// Initialize map
function initMap() {
    map = L.map('map').setView([20.5937, 78.9629], 5); // India center
    
    // OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }).addTo(map);
    
    // Add all layer groups to map
    Object.values(layers).forEach(layer => layer.addTo(map));
    Object.values(suggestionLayers).forEach(layer => layer.addTo(map));
    
    // Initialize drawing controls
    drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    
    let drawControl = new L.Control.Draw({
        draw: {
            polygon: true,
            rectangle: true,
            circle: false,
            marker: false,
            polyline: false,
            circlemarker: false
        },
        edit: {
            featureGroup: drawnItems,
            remove: true
        }
    });
    map.addControl(drawControl);
    
    // Drawing event handlers
    map.on(L.Draw.Event.CREATED, function(e) {
        drawnItems.clearLayers();
        drawnItems.addLayer(e.layer);
        
        const bounds = e.layer.getBounds();
        currentBounds = {
            south: bounds.getSouth(),
            west: bounds.getWest(),
            north: bounds.getNorth(),
            east: bounds.getEast()
        };
        
        runComprehensiveAnalysis();
    });
}

// Run comprehensive analysis
async function runComprehensiveAnalysis() {
    if (!currentBounds) {
        alert('Please draw an area first!');
        return;
    }
    
    // Show loading
    document.getElementById('loadingOverlay').style.display = 'flex';
    
    // Clear previous data
    Object.values(layers).forEach(layer => layer.clearLayers());
    Object.values(suggestionLayers).forEach(layer => layer.clearLayers());
    
    try {
        const response = await fetch('/api/urban-planning/analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ bounds: currentBounds })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayClassification(data);
            displayStatistics(data);
            displayFacilities(data.facilities);
            displaySuggestions(data.suggestions);
            displayRecommendations(data.recommendations);
            
            // Show cards
            document.getElementById('classification-card').style.display = 'block';
            document.getElementById('stats-card').style.display = 'block';
            document.getElementById('recommendations-card').style.display = 'block';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to analyze area. Please try again.');
    } finally {
        document.getElementById('loadingOverlay').style.display = 'none';
    }
}

// Display area classification with detailed information
function displayClassification(data) {
    const content = document.getElementById('classification-content');
    
    let badgeClass = 'bg-success';
    let areaIcon = '🏙️';
    let areaDescription = '';
    let developmentRoadmap = '';
    
    if (data.area_type === 'Urban') {
        badgeClass = 'bg-success';
        areaIcon = '🏙️';
        areaDescription = 'High population density area with extensive infrastructure. Recommendations focus on intensive service coverage.';
    } else if (data.area_type === 'Semi-Urban') {
        badgeClass = 'bg-info';
        areaIcon = '🏘️';
        areaDescription = 'Moderate density area with mixed development. Recommendations balance urban amenities with rural considerations.';
    } else { // Rural
        badgeClass = 'bg-warning';
        areaIcon = '🌾';
        areaDescription = 'Low density area with dispersed population. This area has potential for development to urban standards with proper planning and infrastructure investment.';
        
        // Add development roadmap for rural areas
        developmentRoadmap = `
            <div style="background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%); padding: 20px; border-radius: 12px; margin-top: 20px; color: white; text-align: left; box-shadow: 0 4px 15px rgba(255,94,98,0.3);">
                <h5 style="color: white; border-bottom: 2px solid rgba(255,255,255,0.3); padding-bottom: 10px; margin-bottom: 15px;">
                    🚀 DEVELOPMENT ROADMAP TO URBAN STATUS
                </h5>
                
                <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 8px; margin-bottom: 15px; backdrop-filter: blur(10px);">
                    <strong>📊 Current Status:</strong><br>
                    <small style="line-height: 1.8;">
                    • Classification Score: ${data.classification_score || 'N/A'}/100 (Need 70+ for Urban)<br>
                    • Facility Density: ${(data.facility_density || 0).toFixed(1)}/km² (Need 50+ for Urban)<br>
                    • Housing Density: ${(data.housing_density || 0).toFixed(1)}/km² (Need 100+ for Urban)<br>
                    • Infrastructure Score: ${data.infrastructure_score || 'N/A'}/100 (Need 80+ for Urban)
                    </small>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <strong>⚡ PHASE 1 (0-2 Years): Basic Infrastructure</strong><br>
                    <small style="line-height: 1.8;">
                    ✓ Primary Health Center (1 per 10-15km) - CRITICAL<br>
                    ✓ Food Market/Store (1 per 5-10km) - HIGH<br>
                    ✓ Bus Stop/Transit (Every 10km) - HIGH<br>
                    ✓ Electricity & Water Supply - CRITICAL<br>
                    ✓ Paved Road Connectivity - HIGH
                    </small>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <strong>🏘️ PHASE 2 (2-5 Years): Community Facilities</strong><br>
                    <small style="line-height: 1.8;">
                    ✓ Community Health Center upgrade<br>
                    ✓ 2-3 Grocery Stores - MODERATE<br>
                    ✓ Schools (Primary + Secondary)<br>
                    ✓ Post Office/Bank services<br>
                    ✓ Parks & Waste Management - HIGH
                    </small>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <strong>💼 PHASE 3 (5-10 Years): Economic Development</strong><br>
                    <small style="line-height: 1.8;">
                    ✓ Market Complex for commerce<br>
                    ✓ Small Industries/Processing units<br>
                    ✓ Skill Development Center<br>
                    ✓ High-speed Internet - CRITICAL<br>
                    ✓ Agricultural support facilities
                    </small>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <strong>🏙️ PHASE 4 (10-15 Years): Urbanization</strong><br>
                    <small style="line-height: 1.8;">
                    ✓ Multi-specialty Hospital<br>
                    ✓ 3-5 Supermarkets (modern retail)<br>
                    ✓ Planned Housing (100+ units/km²)<br>
                    ✓ Regular Public Transport (every 2-3km)<br>
                    ✓ Commercial Hub + Industrial Estate
                    </small>
                </div>
                
                <div style="background: rgba(255,255,255,0.2); padding: 12px; border-radius: 8px; border: 2px solid rgba(255,255,255,0.4);">
                    <strong>🎯 Success Factors:</strong><br>
                    <small style="line-height: 1.8;">
                    • Population Growth: 3-5x increase needed<br>
                    • Investment: ₹50-100 crore over 10 years<br>
                    • Jobs: 500-1000 to retain population<br>
                    • Timeline: Services < 30 min travel time
                    </small>
                </div>
                
                <div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.25); border-radius: 8px; text-align: center;">
                    <small><strong>💡 RED DOTS below show PHASE 1 priority locations!</strong></small>
                </div>
            </div>
        `;
    }
    
    content.innerHTML = `
        <div class="text-center">
            <div style="font-size: 48px; margin-bottom: 10px;">${areaIcon}</div>
            <span class="badge ${badgeClass} mb-3" style="font-size: 20px; padding: 10px 20px;">
                ${data.area_type} Area
            </span>
            <p style="margin-top: 15px;"><strong>Area Size:</strong> ${data.area_km2} km²</p>
            <p><strong>Facility Density:</strong> ${(data.facility_density || 0).toFixed(1)} per km²</p>
            ${data.classification_score ? `<p><strong>Classification Score:</strong> ${data.classification_score}/100</p>` : ''}
            <div style="background: rgba(74, 144, 226, 0.2); padding: 15px; border-radius: 8px; margin-top: 15px; text-align: left; border: 1px solid rgba(74, 144, 226, 0.4);">
                <strong style="color: #ffffff; font-size: 16px;">📊 Area Characteristics:</strong><br>
                <small style="line-height: 1.8; color: #ffffff; font-weight: 500;">${areaDescription}</small>
            </div>
            <div style="background: rgba(233, 69, 96, 0.25); padding: 12px; border-radius: 8px; margin-top: 10px; border: 2px solid rgba(233, 69, 96, 0.5);">
                <strong style="color: #ffffff; font-size: 15px;">🔴 Red Dots:</strong> <span style="color: #ffffff; font-weight: 500;">Suggested facility locations optimized for ${data.area_type.toLowerCase()} coverage</span>
            </div>
            ${developmentRoadmap}
        </div>
    `;
}

// Display statistics
function displayStatistics(data) {
    const content = document.getElementById('stats-content');
    
    const statHtml = Object.entries(data.counts).map(([key, value]) => {
        const color = layerColors[key] || '#6c757d';
        return `
            <div class="stat-card" style="background: ${color};">
                <div class="stat-number">${value}</div>
                <div class="stat-label">${key.charAt(0).toUpperCase() + key.slice(1)}</div>
            </div>
        `;
    }).join('');
    
    content.innerHTML = statHtml;
}

// Display facilities on map with improved labels
function displayFacilities(facilities) {
    Object.entries(facilities).forEach(([category, locations]) => {
        const color = layerColors[category];
        const layer = layers[category];
        
        locations.forEach(loc => {
            const marker = L.circleMarker([loc.lat, loc.lon], {
                radius: 6,
                fillColor: color,
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            });
            
            // Create detailed popup with all available information
            const facilityType = loc.type || category;
            const address = loc.address ? loc.address.trim() : 'Address not available';
            
            marker.bindPopup(`
                <div style="min-width: 200px;">
                    <strong style="color: ${color}; font-size: 14px;">
                        ${category.charAt(0).toUpperCase() + category.slice(1)}
                    </strong><br>
                    <strong>${loc.name}</strong><br>
                    <small style="color: #666;">Type: ${facilityType}</small><br>
                    ${address !== 'Address not available' ? `<small>${address}</small><br>` : ''}
                    <small>📍 ${loc.lat.toFixed(5)}, ${loc.lon.toFixed(5)}</small>
                </div>
            `);
            
            // Add tooltip for quick view
            marker.bindTooltip(loc.name, {
                permanent: false,
                direction: 'top',
                className: 'facility-tooltip'
            });
            
            marker.addTo(layer);
        });
        
        // Update count
        document.getElementById(`count-${category}`).textContent = locations.length;
    });
}

// Display RED DOT suggestions with detailed information
function displaySuggestions(suggestions) {
    Object.entries(suggestions).forEach(([category, locations]) => {
        const layer = suggestionLayers[category];
        
        locations.forEach((loc, index) => {
            // Create SOLID red marker (no pulsing, no bouncing)
            // PRECISE MARKER: Requirement-based optimal location
            const marker = L.circleMarker([loc.lat, loc.lon], {
                radius: 7,  // Slightly larger for visibility
                fillColor: '#ff0000',
                color: '#ffffff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.95
            });
            
            // Enhanced popup with REQUIREMENT-BASED scoring details
            const coverageInfo = loc.avg_distance 
                ? `<small style="color: #ffffff;">📏 Coverage: ${loc.avg_distance} km avg</small><br>`
                : '';
            
            const scoreInfo = loc.score 
                ? `<small style="color: #ffffff;">🎯 Score: ${loc.score} (optimized)</small><br>`
                : '';
            
            const distanceInfo = loc.distance && loc.distance !== 999
                ? `<small style="color: #ffffff;">📍 Service gap: ${loc.distance} km</small><br>`
                : '';
            
            const populationInfo = loc.population_score
                ? `<small style="color: #ffffff;">👥 Population: ${loc.population_score}/20</small><br>`
                : '';
            
            const accessInfo = loc.accessibility_score
                ? `<small style="color: #ffffff;">🚌 Transport: ${loc.accessibility_score}/15</small><br>`
                : '';
            
            marker.bindPopup(`
                <div style="text-align: center; min-width: 280px;">
                    <div style="background: linear-gradient(135deg, #ff0000, #cc0000); color: white; padding: 12px; margin: -10px -10px 10px -10px; border-radius: 5px 5px 0 0; box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
                        <strong style="font-size: 17px;">🎯 OPTIMAL LOCATION</strong>
                    </div>
                    <div style="background: linear-gradient(135deg, #fff5f5, #ffe0e0); padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                        <strong style="color: #ff0000; font-size: 16px;">
                            ${category.charAt(0).toUpperCase() + category.slice(1)} Facility
                        </strong><br>
                        <span style="background: #ff0000; color: white; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; display: inline-block; margin-top: 5px;">
                            PRIORITY #${index + 1}
                        </span>
                    </div>
                    <div style="text-align: left; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; margin-bottom: 8px;">
                        <strong style="color: #cc0000; font-size: 13px;">📊 Location Analysis:</strong><br>
                        ${distanceInfo}
                        ${coverageInfo}
                        ${populationInfo}
                        ${accessInfo}
                        ${scoreInfo}
                    </div>
                    <small style="color: #999; font-size: 11px;">📍 ${loc.lat}, ${loc.lon}</small>
                    <div style="background: linear-gradient(135deg, #ffe0e0, #ffcccc); padding: 10px; margin-top: 10px; border-radius: 8px; border-left: 4px solid #ff0000;">
                        <strong style="color: #cc0000; font-size: 13px;">💡 Why Here?</strong><br>
                        <span style="color: #666; font-size: 12px;">Strategically placed based on:<br>
                        ✓ Service coverage gaps<br>
                        ✓ Population proximity<br>
                        ✓ Transport accessibility<br>
                        ✓ Optimal facility spacing</span>
                    </div>
                </div>
            `);
            
            // Enhanced tooltip with priority
            marker.bindTooltip(`🔴 ${category} #${index + 1} (Score: ${loc.score || 'N/A'})`, {
                permanent: false,
                direction: 'top',
                className: 'suggestion-tooltip',
                offset: [0, -10],
                opacity: 0.95
            });
            
            marker.addTo(layer);
        });
        
        // Update suggestion count
        document.getElementById(`suggest-count-${category}`).textContent = locations.length;
    });
}

// Display AI recommendations
function displayRecommendations(recommendations) {
    const accordion = document.getElementById('recommendationsAccordion');
    
    const objectiveOrder = [
        { key: 'food_access', title: 'Food Access', icon: 'utensils' },
        { key: 'housing', title: 'Housing', icon: 'home' },
        { key: 'transportation', title: 'Transportation', icon: 'bus' },
        { key: 'pollution', title: 'Pollution Control', icon: 'smog' },
        { key: 'healthcare', title: 'Healthcare', icon: 'hospital' },
        { key: 'parks', title: 'Parks & Recreation', icon: 'tree' },
        { key: 'growth', title: 'Urban Growth', icon: 'chart-line' },
        { key: 'public_health', title: 'Public Health', icon: 'shield-virus' },
        { key: 'environment', title: 'Environment Protection', icon: 'leaf' },
        { key: 'agriculture', title: 'Agriculture', icon: 'tractor' },
        { key: 'waste', title: 'Waste Management', icon: 'recycle' },
        { key: 'energy', title: 'Energy Access', icon: 'bolt' }
    ];
    
    accordion.innerHTML = objectiveOrder.map((obj, index) => {
        const rec = recommendations[obj.key];
        if (!rec) return '';
        
        let urgencyClass = 'urgency-low';
        if (rec.urgency === 'Critical') urgencyClass = 'urgency-critical';
        else if (rec.urgency === 'High') urgencyClass = 'urgency-high';
        else if (rec.urgency === 'Moderate') urgencyClass = 'urgency-moderate';
        
        return `
            <div class="accordion-item">
                <h2 class="accordion-header" id="heading${index}">
                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                            data-bs-target="#collapse${index}" aria-expanded="false" aria-controls="collapse${index}">
                        <i class="fas fa-${obj.icon} me-2"></i> ${obj.title}
                        ${rec.urgency ? `<span class="badge ${urgencyClass} ms-2">${rec.urgency}</span>` : ''}
                    </button>
                </h2>
                <div id="collapse${index}" class="accordion-collapse collapse" 
                     aria-labelledby="heading${index}" data-bs-parent="#recommendationsAccordion">
                    <div class="accordion-body">
                        ${formatRecommendation(rec)}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Format recommendation content
function formatRecommendation(rec) {
    let html = '';
    
    if (rec.status) html += `<p style="color: #ffffff; font-weight: 500;"><strong style="color: #a0d2ff;">Status:</strong> ${rec.status}</p>`;
    if (rec.additional_needed !== undefined) html += `<p style="color: #ffffff; font-weight: 500;"><strong style="color: #a0d2ff;">Additional Needed:</strong> ${rec.additional_needed}</p>`;
    if (rec.facilities_needed !== undefined) html += `<p style="color: #ffffff; font-weight: 500;"><strong style="color: #a0d2ff;">Facilities Needed:</strong> ${rec.facilities_needed}</p>`;
    if (rec.new_units_needed !== undefined) html += `<p style="color: #ffffff; font-weight: 500;"><strong style="color: #a0d2ff;">New Units Needed:</strong> ${rec.new_units_needed}</p>`;
    if (rec.additional_stops_needed !== undefined) html += `<p style="color: #ffffff; font-weight: 500;"><strong style="color: #a0d2ff;">Stops Needed:</strong> ${rec.additional_stops_needed}</p>`;
    if (rec.additional_facilities !== undefined) html += `<p style="color: #ffffff; font-weight: 500;"><strong style="color: #a0d2ff;">Additional Facilities:</strong> ${rec.additional_facilities}</p>`;
    if (rec.additional_infrastructure !== undefined) html += `<p style="color: #ffffff; font-weight: 500;"><strong style="color: #a0d2ff;">Infrastructure Needed:</strong> ${rec.additional_infrastructure}</p>`;
    if (rec.risk_level) html += `<p style="color: #ffffff; font-weight: 500;"><strong style="color: #a0d2ff;">Risk Level:</strong> ${rec.risk_level}</p>`;
    if (rec.priority) html += `<p style="color: #ffffff; font-weight: 500;"><strong style="color: #a0d2ff;">Priority:</strong> ${rec.priority}</p>`;
    if (rec.recommendation) html += `<p style="color: #ffffff; font-weight: 500; line-height: 1.8;"><strong style="color: #a0d2ff;">Recommendation:</strong> ${rec.recommendation}</p>`;
    
    return html || '<p style="color: #c0d5f0;">No specific recommendations at this time.</p>';
}

// Toggle layer visibility
function toggleLayer(category) {
    const checkbox = document.getElementById(`layer-${category}`);
    const layer = layers[category];
    
    if (checkbox.checked) {
        map.addLayer(layer);
    } else {
        map.removeLayer(layer);
    }
}

// Toggle suggestion visibility
function toggleSuggestions(category) {
    const checkbox = document.getElementById(`suggest-${category}`);
    const layer = suggestionLayers[category];
    
    if (checkbox.checked) {
        map.addLayer(layer);
    } else {
        map.removeLayer(layer);
    }
}

// Toggle all layers
function toggleAllLayers() {
    const allCheckboxes = document.querySelectorAll('#filter-panel input[type="checkbox"]');
    const firstChecked = allCheckboxes[0].checked;
    
    allCheckboxes.forEach(checkbox => {
        checkbox.checked = !firstChecked;
        const event = new Event('change');
        checkbox.dispatchEvent(event);
    });
}

// Clear all data
function clearAllData() {
    if (confirm('Are you sure you want to clear all data?')) {
        // Clear layers
        Object.values(layers).forEach(layer => layer.clearLayers());
        Object.values(suggestionLayers).forEach(layer => layer.clearLayers());
        
        // Clear drawn items
        drawnItems.clearLayers();
        
        // Hide cards
        document.getElementById('classification-card').style.display = 'none';
        document.getElementById('stats-card').style.display = 'none';
        document.getElementById('recommendations-card').style.display = 'none';
        
        // Reset counts
        Object.keys(layers).forEach(category => {
            document.getElementById(`count-${category}`).textContent = '0';
        });
        
        Object.keys(suggestionLayers).forEach(category => {
            document.getElementById(`suggest-count-${category}`).textContent = '0';
        });
        
        currentBounds = null;
    }
}

// Initialize on page load
window.addEventListener('load', initMap);
