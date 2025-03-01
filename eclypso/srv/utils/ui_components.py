import gradio as gr
import logging
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import asyncio
import time
from PIL import Image

logger = logging.getLogger('biomedparse_server')

def create_image_viewer_components():
    """
    Create components for the image viewer interface.
    
    Returns:
        Tuple of (components, states) where:
        - components is a dict of UI components
        - states is a dict of state components
    """
    components = {}
    states = {}
    
    # Create slider for image navigation
    components['slider'] = gr.Slider(
        minimum=0, 
        maximum=1000,  # Increased from 100 to 1000 to handle larger series
        step=1, 
        value=0, 
        label="Image Index",
        interactive=True
    )
    
    # Create playback control buttons
    with gr.Row(elem_classes="controls-row"):
        components['prev_button'] = gr.Button("⏮️ Previous", size="sm")
        components['play_button'] = gr.Button("▶️ Play", size="sm", variant="primary")
        components['stop_button'] = gr.Button("⏹️ Stop", size="sm")
        components['next_button'] = gr.Button("⏭️ Next", size="sm")
    
    # Create playback settings
    with gr.Row():
        with gr.Column(scale=2, elem_classes="speed-slider-container"):
            components['speed_slider'] = gr.Slider(
                minimum=1,
                maximum=30,
                step=1,
                value=10,
                label="Frames per second",
                interactive=True
            )
        with gr.Column(scale=1):
            components['loop_checkbox'] = gr.Checkbox(
                value=True, 
                label="Loop playback", 
                interactive=True
            )
    
    # Create status indicators
    with gr.Row():
        with gr.Column(scale=2):
            components['playback_indicator'] = gr.HTML(
                value="<span style='color:gray'>Stopped</span>", 
                label="Status"
            )
        with gr.Column(scale=1):
            components['image_counter'] = gr.HTML(
                value="Image: 0 / 0", 
                label=""
            )
    
    # Create progress bar for loading
    components['progress_bar'] = gr.Progress()
    
    # Create image display using HTML instead of Gallery for smoother transitions
    components['image'] = gr.HTML(
        value="""
        <div class="custom-image-viewer">
            <div class="image-container" style="height: 512px; display: flex; justify-content: center; align-items: center;">
                <img id="current-image" src="" alt="No image loaded" style="max-height: 100%; max-width: 100%; opacity: 1; transition: opacity 0.2s ease-in-out;">
                <img id="next-image" src="" alt="" style="max-height: 100%; max-width: 100%; position: absolute; opacity: 0; transition: opacity 0.2s ease-in-out;">
            </div>
        </div>
        """,
        label="Current Image"
    )
    
    # Create results display
    components['results'] = gr.Textbox(
        label="Results", 
        show_label=True, 
        lines=5
    )
    
    # Create state components
    states['images'] = gr.State([])
    states['playback'] = gr.State({"playing": False, "interval_id": None})
    states['preload'] = gr.State({"preloaded": False, "preloaded_indices": []})
    states['cache_info'] = gr.State({"total": 0, "loaded": 0, "memory_mb": 0})
    
    return components, states

def setup_playback_controls(components, states):
    """
    Set up event handlers for playback controls.
    
    Args:
        components: Dict of UI components
        states: Dict of state components
        
    Returns:
        None
    """
    # Start playback
    components['play_button'].click(
        fn=lambda: {"playing": True, "interval_id": None},
        inputs=[],
        outputs=[states['playback']]
    ).then(
        fn=lambda fps: f"<div id='js-trigger' data-action='startPlayback' data-fps='{fps}'><span style='color:green'>Playing at {fps} fps</span></div>",
        inputs=[components['speed_slider']],
        outputs=[components['playback_indicator']]
    ).then(
        # Add an additional callback to ensure the JavaScript is triggered
        fn=lambda: None,  # This is just a dummy function to add a delay
        inputs=[],
        outputs=[]
    )
    
    # Stop playback
    components['stop_button'].click(
        fn=lambda: {"playing": False, "interval_id": None},
        inputs=[],
        outputs=[states['playback']]
    ).then(
        fn=lambda: "<div id='js-trigger' data-action='stopPlayback'><span style='color:gray'>Stopped</span></div>",
        inputs=[],
        outputs=[components['playback_indicator']]
    )
    
    # Previous image
    components['prev_button'].click(
        fn=show_preloading,
        inputs=[],
        outputs=[components['playback_indicator']]
    ).then(
        fn=navigate_to_prev_image,
        inputs=[components['slider'], states['images'], states['preload']],
        outputs=[states['playback'], components['slider'], components['image'], 
                components['image_counter'], components['playback_indicator'], states['preload']]
    )
    
    # Next image
    components['next_button'].click(
        fn=show_preloading,
        inputs=[],
        outputs=[components['playback_indicator']]
    ).then(
        fn=navigate_to_next_image,
        inputs=[components['slider'], states['images'], states['preload']],
        outputs=[states['playback'], components['slider'], components['image'], 
                components['image_counter'], components['playback_indicator'], states['preload']]
    )
    
    # Slider change
    components['slider'].change(
        fn=show_preloading,
        inputs=[],
        outputs=[components['playback_indicator']]
    ).then(
        fn=update_image_from_slider,
        inputs=[components['slider'], states['images']],
        outputs=[components['image'], components['image_counter']]
    ).then(
        fn=preload_adjacent_images,
        inputs=[states['images'], components['slider'], states['preload']],
        outputs=[states['images'], states['preload'], components['playback_indicator']]
    )
    
    # Speed slider change
    components['speed_slider'].change(
        fn=update_playback_speed,
        inputs=[components['speed_slider'], states['playback']],
        outputs=[components['speed_slider'], states['playback']]
    ).then(
        fn=lambda fps, status: f"<div id='js-trigger' data-action='updatePlaybackSpeed' data-fps='{fps}'><span style='color:green'>Updated to {fps} fps</span></div>" if status and status.get("playing", False) else "<span style='color:gray'>Speed updated</span>",
        inputs=[components['speed_slider'], states['playback']],
        outputs=[components['playback_indicator']]
    )

def show_preloading():
    """Show preloading indicator"""
    return "<div id='preload-trigger' data-action='showPreloading'><span style='color:gray'>Loading...</span></div>"

def navigate_to_prev_image(current_idx, images, preload_state):
    """Navigate to the previous image"""
    # Extract the current index value, handling both int and dict cases
    idx_value = int(current_idx) if isinstance(current_idx, (int, float)) else current_idx["value"] if isinstance(current_idx, dict) and "value" in current_idx else 0
    new_idx = max(0, idx_value - 1)
    
    # Get the image at the new index
    next_image = images[new_idx] if images and 0 <= new_idx < len(images) else None
    
    # Update preload state
    preload_indices = [new_idx]
    if new_idx + 1 < len(images):
        preload_indices.append(new_idx + 1)
    if new_idx - 1 >= 0:
        preload_indices.append(new_idx - 1)
    
    already_preloaded = set(preload_state.get("preloaded_indices", []))
    for idx in preload_indices:
        if idx not in already_preloaded and 0 <= idx < len(images):
            # Access the image to ensure it's loaded
            _ = images[idx]
            already_preloaded.add(idx)
    
    updated_preload_state = {"preloaded": True, "preloaded_indices": list(already_preloaded)}
    
    # For HTML component, we need to create HTML with the image embedded
    if next_image is not None:
        # Convert PIL image to base64 for embedding in HTML
        import base64
        import io
        
        # Save image to bytes buffer
        buffer = io.BytesIO()
        next_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create HTML with embedded image and transition effect
        html = f"""
        <div class="custom-image-viewer">
            <div class="image-container" style="height: 512px; display: flex; justify-content: center; align-items: center;">
                <img id="current-image" src="data:image/png;base64,{img_str}" alt="Image {new_idx + 1}" 
                     style="max-height: 100%; max-width: 100%; opacity: 1; transition: opacity 0.2s ease-in-out;">
                <img id="next-image" src="" alt="" 
                     style="max-height: 100%; max-width: 100%; position: absolute; opacity: 0; transition: opacity 0.2s ease-in-out;">
            </div>
        </div>
        <div id="js-trigger" data-action="imageLoaded" data-index="{new_idx}" data-direction="prev"></div>
        """
    else:
        html = """
        <div class="custom-image-viewer">
            <div class="image-container" style="height: 512px; display: flex; justify-content: center; align-items: center;">
                <div style="color: gray;">No image available</div>
            </div>
        </div>
        """
    
    return (
        {"playing": False, "interval_id": None},  # Stop playback
        {"value": new_idx},  # Update slider value
        html,  # Update image HTML
        f"Image: {new_idx + 1} / {len(images)}" if images else "Image: 0 / 0",  # Update counter
        "<span style='color:gray'>Stopped</span>",  # Update playback indicator
        updated_preload_state  # Update preload state
    )

def navigate_to_next_image(current_idx, images, preload_state):
    """Navigate to the next image"""
    # Extract the current index value, handling both int and dict cases
    idx_value = int(current_idx) if isinstance(current_idx, (int, float)) else current_idx["value"] if isinstance(current_idx, dict) and "value" in current_idx else 0
    max_idx = len(images) - 1 if images else 0
    new_idx = min(max_idx, idx_value + 1)
    
    # Get the image at the new index
    next_image = images[new_idx] if images and 0 <= new_idx < len(images) else None
    
    # Update preload state
    preload_indices = [new_idx]
    if new_idx + 1 < len(images):
        preload_indices.append(new_idx + 1)
    if new_idx - 1 >= 0:
        preload_indices.append(new_idx - 1)
    
    already_preloaded = set(preload_state.get("preloaded_indices", []))
    for idx in preload_indices:
        if idx not in already_preloaded and 0 <= idx < len(images):
            # Access the image to ensure it's loaded
            _ = images[idx]
            already_preloaded.add(idx)
    
    updated_preload_state = {"preloaded": True, "preloaded_indices": list(already_preloaded)}
    
    # For HTML component, we need to create HTML with the image embedded
    if next_image is not None:
        # Convert PIL image to base64 for embedding in HTML
        import base64
        import io
        
        # Save image to bytes buffer
        buffer = io.BytesIO()
        next_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create HTML with embedded image and transition effect
        html = f"""
        <div class="custom-image-viewer">
            <div class="image-container" style="height: 512px; display: flex; justify-content: center; align-items: center;">
                <img id="current-image" src="data:image/png;base64,{img_str}" alt="Image {new_idx + 1}" 
                     style="max-height: 100%; max-width: 100%; opacity: 1; transition: opacity 0.2s ease-in-out;">
                <img id="next-image" src="" alt="" 
                     style="max-height: 100%; max-width: 100%; position: absolute; opacity: 0; transition: opacity 0.2s ease-in-out;">
            </div>
        </div>
        <div id="js-trigger" data-action="imageLoaded" data-index="{new_idx}" data-direction="next"></div>
        """
    else:
        html = """
        <div class="custom-image-viewer">
            <div class="image-container" style="height: 512px; display: flex; justify-content: center; align-items: center;">
                <div style="color: gray;">No image available</div>
            </div>
        </div>
        """
    
    return (
        {"playing": False, "interval_id": None},  # Stop playback
        {"value": new_idx},  # Update slider value
        html,  # Update image HTML
        f"Image: {new_idx + 1} / {len(images)}" if images else "Image: 0 / 0",  # Update counter
        "<span style='color:gray'>Stopped</span>",  # Update playback indicator
        updated_preload_state  # Update preload state
    )

def update_image_from_slider(idx, images):
    """Update the displayed image based on slider value"""
    # Extract the index value
    idx_value = int(idx) if isinstance(idx, (int, float)) else idx["value"] if isinstance(idx, dict) and "value" in idx else 0
    
    # Get the image at this index
    image = images[idx_value] if images and 0 <= idx_value < len(images) else None
    
    # Update counter text
    counter_text = f"Image: {idx_value + 1} / {len(images)}" if images else "Image: 0 / 0"
    
    # For HTML component, we need to create HTML with the image embedded
    if image is not None:
        # Convert PIL image to base64 for embedding in HTML
        import base64
        import io
        
        # Save image to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create HTML with embedded image
        html = f"""
        <div class="custom-image-viewer">
            <div class="image-container" style="height: 512px; display: flex; justify-content: center; align-items: center;">
                <img id="current-image" src="data:image/png;base64,{img_str}" alt="Image {idx_value + 1}" 
                     style="max-height: 100%; max-width: 100%; opacity: 1; transition: opacity 0.2s ease-in-out;">
                <img id="next-image" src="" alt="" 
                     style="max-height: 100%; max-width: 100%; position: absolute; opacity: 0; transition: opacity 0.2s ease-in-out;">
            </div>
        </div>
        <div id="js-trigger" data-action="imageLoaded" data-index="{idx_value}"></div>
        """
    else:
        html = """
        <div class="custom-image-viewer">
            <div class="image-container" style="height: 512px; display: flex; justify-content: center; align-items: center;">
                <div style="color: gray;">No image available</div>
            </div>
        </div>
        """
    
    return html, counter_text

def preload_adjacent_images(images, current_idx, preload_state):
    """Preload adjacent images to improve performance"""
    if not images or len(images) <= 1:
        return images, preload_state, ""
    
    # Current index as integer
    idx = int(current_idx) if isinstance(current_idx, (int, float)) else current_idx.get("value", 0) if isinstance(current_idx, dict) else 0
    
    # Calculate indices to preload (current, next, previous, next+1, previous-1)
    preload_indices = [idx]
    if idx + 1 < len(images):
        preload_indices.append(idx + 1)
    if idx - 1 >= 0:
        preload_indices.append(idx - 1)
    if idx + 2 < len(images):
        preload_indices.append(idx + 2)
    if idx - 2 >= 0:
        preload_indices.append(idx - 2)
    
    # Check if these indices are already preloaded
    already_preloaded = set(preload_state.get("preloaded_indices", []))
    indices_to_preload = [i for i in preload_indices if i not in already_preloaded]
    
    if not indices_to_preload:
        # All needed images are already preloaded
        return images, preload_state, ""
    
    logger.info(f"Preloading images at indices: {indices_to_preload}")
    
    # Update preload state to indicate we're preloading
    preload_state = {"preloaded": True, "preloaded_indices": list(already_preloaded)}
    
    # Force images to be loaded into memory
    for i in indices_to_preload:
        if 0 <= i < len(images):
            # Access the image to ensure it's loaded
            _ = images[i]
            preload_state["preloaded_indices"].append(i)
    
    # Create a visual indicator for preloading
    preload_indicator = "<div id='preload-trigger' data-action='hidePreloading'><span style='color:gray'>Loaded</span></div>"
    
    return images, preload_state, preload_indicator

def update_playback_speed(fps, playback_state):
    """Update playback speed"""
    # Convert fps to interval in seconds
    interval = 1.0 / float(fps)
    logger.info(f"Updating timer interval: {interval}s ({fps} fps)")
    
    # We can't directly update the timer interval in this version of Gradio
    # Instead, we'll use a workaround by stopping and starting playback if it's currently playing
    is_playing = playback_state.get("playing", False) if playback_state else False
    
    if is_playing:
        logger.info("Playback is active, will restart with new interval")
        # Return the same playback state to maintain playback
        return fps, playback_state
    else:
        logger.info("Playback is not active, just updating FPS value")
        # Return the same playback state
        return fps, playback_state

def auto_advance(playback_status, current_idx, images, fps, should_loop, preload_state):
    """Auto-advance function for playback"""
    # Add debug logging
    logger.info(f"Auto advance called: playing={playback_status.get('playing', False)}, idx={current_idx}, images={len(images) if images else 0}, fps={fps}")
    
    if not playback_status or not playback_status.get("playing", False):
        logger.info("Playback not active, skipping auto-advance")
        return playback_status, current_idx, None, "", "", preload_state
    
    # Extract the current index value
    idx_value = int(current_idx) if isinstance(current_idx, (int, float)) else current_idx["value"] if isinstance(current_idx, dict) and "value" in current_idx else 0
    max_idx = len(images) - 1 if images else 0
    
    logger.info(f"Current index: {idx_value}, Max index: {max_idx}")
    
    # Calculate next index
    next_idx = idx_value + 1
    
    # Check if we reached the end
    if next_idx > max_idx:
        if should_loop:
            next_idx = 0  # Loop back to beginning
            logger.info(f"Reached end, looping back to beginning: {next_idx}")
        else:
            # Stop playback if we're not looping
            logger.info("Reached end, stopping playback (no loop)")
            return {"playing": False, "interval_id": None}, current_idx, None, "", "<span style='color:gray'>Stopped</span>", preload_state
    
    # Get the next image
    next_image = images[next_idx] if images and 0 <= next_idx < len(images) else None
    
    # Update counter text
    counter_text = f"Image: {next_idx + 1} / {len(images)}" if images else "Image: 0 / 0"
    
    # Calculate the appropriate delay based on fps
    # This is used in the status text only, the actual timing is controlled by the Timer component
    fps_value = float(fps) if isinstance(fps, (int, float)) else fps.get("value", 10) if isinstance(fps, dict) else 10
    
    # Update status text
    status_text = f"<span style='color:green'>Playing at {fps_value} fps" + (" (looping)" if should_loop else " (no loop)") + "</span>"
    
    # Preload next images
    preload_indices = [next_idx]
    if next_idx + 1 <= max_idx:
        preload_indices.append(next_idx + 1)
    if next_idx + 2 <= max_idx:
        preload_indices.append(next_idx + 2)
    
    # Update preload state
    already_preloaded = set(preload_state.get("preloaded_indices", []))
    for idx in preload_indices:
        if idx not in already_preloaded and 0 <= idx < len(images):
            # Access the image to ensure it's loaded
            _ = images[idx]
            already_preloaded.add(idx)
    
    preload_state = {"preloaded": True, "preloaded_indices": list(already_preloaded)}
    
    # For HTML component, we need to create HTML with the image embedded
    if next_image is not None:
        # Convert PIL image to base64 for embedding in HTML
        import base64
        import io
        
        # Save image to bytes buffer
        buffer = io.BytesIO()
        next_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create HTML with embedded image and transition effect
        html = f"""
        <div class="custom-image-viewer">
            <div class="image-container" style="height: 512px; display: flex; justify-content: center; align-items: center;">
                <img id="current-image" src="data:image/png;base64,{img_str}" alt="Image {next_idx + 1}" 
                     style="max-height: 100%; max-width: 100%; opacity: 1; transition: opacity 0.2s ease-in-out;">
                <img id="next-image" src="" alt="" 
                     style="max-height: 100%; max-width: 100%; position: absolute; opacity: 0; transition: opacity 0.2s ease-in-out;">
            </div>
        </div>
        <div id="js-trigger" data-action="imageLoaded" data-index="{next_idx}" data-direction="auto"></div>
        """
    else:
        html = """
        <div class="custom-image-viewer">
            <div class="image-container" style="height: 512px; display: flex; justify-content: center; align-items: center;">
                <div style="color: gray;">No image available</div>
            </div>
        </div>
        """
    
    logger.info(f"Moving to next image: {next_idx}")
    return {"playing": True, "interval_id": None}, {"value": next_idx}, html, counter_text, status_text, preload_state

def get_javascript_for_image_viewer():
    """Get JavaScript code for image viewer"""
    return """
    // Playback control functions
    function setupPlaybackControl() {
        // Store the interval ID globally
        if (typeof window.dicomPlayerInterval === 'undefined') {
            window.dicomPlayerInterval = null;
        }
        
        // Create an image cache for preloading
        if (typeof window.imageCache === 'undefined') {
            window.imageCache = {};
        }
        
        // Function to start playback
        window.startPlayback = function(fps) {
            // Clear any existing interval
            if (window.dicomPlayerInterval) {
                clearInterval(window.dicomPlayerInterval);
            }
            
            // Calculate interval in milliseconds
            const interval = 1000 / fps;
            
            // Set up a new interval that advances the slider
            window.dicomPlayerInterval = setInterval(() => {
                // Find the slider - try multiple selectors to be more robust
                const slider = document.querySelector('input[aria-label="Image Index"]') || 
                               document.querySelector('.player-container input[type="range"]') ||
                               document.querySelector('.gradio-slider input[type="range"]');
                
                if (slider) {
                    // Get current value and max value
                    const currentValue = parseInt(slider.value);
                    const maxValue = parseInt(slider.max);
                    
                    // Calculate next value (with looping)
                    const loopCheckbox = document.querySelector('input[aria-label="Loop playback"]') ||
                                         document.querySelector('.player-container input[type="checkbox"]');
                    const shouldLoop = loopCheckbox && loopCheckbox.checked;
                    
                    let nextValue = currentValue + 1;
                    if (nextValue > maxValue) {
                        if (shouldLoop) {
                            nextValue = 0;
                        } else {
                            // Stop playback if we reached the end and not looping
                            window.stopPlayback();
                            
                            // Find and click the stop button to update UI
                            const stopButton = document.querySelector('.controls-row button:nth-child(3)');
                            if (stopButton) {
                                stopButton.click();
                            }
                            return;
                        }
                    }
                    
                    // Update slider value
                    slider.value = nextValue;
                    
                    // Dispatch change event to trigger Gradio's event handlers
                    const event = new Event('change', { bubbles: true });
                    slider.dispatchEvent(event);
                    
                    // Also dispatch input event for good measure
                    const inputEvent = new Event('input', { bubbles: true });
                    slider.dispatchEvent(inputEvent);
                    
                    // Log for debugging
                    console.log(`Advanced to frame ${nextValue} of ${maxValue}`);
                } else {
                    console.warn('Slider not found for playback. Trying to find it by other means...');
                    // Try to find any slider in the document
                    const allSliders = document.querySelectorAll('input[type="range"]');
                    console.log(`Found ${allSliders.length} sliders in the document`);
                    
                    // If we found any sliders, use the first one
                    if (allSliders.length > 0) {
                        const firstSlider = allSliders[0];
                        console.log(`Using first slider found: ${firstSlider.getAttribute('aria-label') || 'unnamed'}`);
                        
                        // Get current value and max value
                        const currentValue = parseInt(firstSlider.value);
                        const maxValue = parseInt(firstSlider.max);
                        
                        // Calculate next value
                        let nextValue = currentValue + 1;
                        if (nextValue > maxValue) nextValue = 0;
                        
                        // Update slider value
                        firstSlider.value = nextValue;
                        
                        // Dispatch events
                        firstSlider.dispatchEvent(new Event('change', { bubbles: true }));
                        firstSlider.dispatchEvent(new Event('input', { bubbles: true }));
                        
                        console.log(`Advanced to frame ${nextValue} using fallback method`);
                    }
                }
            }, interval);
            
            console.log(`Started playback at ${fps} FPS (${interval}ms interval)`);
        };
        
        // Function to stop playback
        window.stopPlayback = function() {
            if (window.dicomPlayerInterval) {
                clearInterval(window.dicomPlayerInterval);
                window.dicomPlayerInterval = null;
                console.log('Stopped playback');
            }
        };
        
        // Function to update playback speed
        window.updatePlaybackSpeed = function(fps) {
            if (window.dicomPlayerInterval) {
                // Restart with new speed
                window.stopPlayback();
                window.startPlayback(fps);
                console.log(`Updated playback speed to ${fps} FPS`);
            }
        };
        
        // Function to hide loading indicators for the speed slider
        window.hideSpeedSliderLoading = function() {
            // Find all loading indicators in the speed slider container
            const loadingIndicators = document.querySelectorAll('.speed-slider-container .progress-container, .speed-slider-container .progress, .speed-slider-container .progress-bar');
            loadingIndicators.forEach(indicator => {
                indicator.style.display = 'none';
                indicator.style.opacity = '0';
                indicator.style.visibility = 'hidden';
            });
            console.log('Hiding speed slider loading indicators');
        };
        
        // Set up a mutation observer to watch for trigger elements
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    // Look for our trigger elements
                    const triggers = document.querySelectorAll('#js-trigger');
                    triggers.forEach(trigger => {
                        const action = trigger.getAttribute('data-action');
                        console.log(`Found trigger with action: ${action}`);
                        
                        if (action === 'startPlayback') {
                            const fps = parseFloat(trigger.getAttribute('data-fps') || '10');
                            console.log(`Starting playback at ${fps} FPS`);
                            window.startPlayback(fps);
                            
                            // Remove the trigger to avoid duplicate calls
                            trigger.removeAttribute('id');
                            trigger.setAttribute('data-processed', 'true');
                        } else if (action === 'stopPlayback') {
                            console.log('Stopping playback');
                            window.stopPlayback();
                            
                            // Remove the trigger to avoid duplicate calls
                            trigger.removeAttribute('id');
                            trigger.setAttribute('data-processed', 'true');
                        } else if (action === 'updatePlaybackSpeed') {
                            const fps = parseFloat(trigger.getAttribute('data-fps') || '10');
                            console.log(`Updating playback speed to ${fps} FPS`);
                            window.updatePlaybackSpeed(fps);
                            
                            // Remove the trigger to avoid duplicate calls
                            trigger.removeAttribute('id');
                            trigger.setAttribute('data-processed', 'true');
                        } else if (action === 'imageLoaded') {
                            // Handle image loaded event for smooth transitions
                            const index = parseInt(trigger.getAttribute('data-index') || '0');
                            const direction = trigger.getAttribute('data-direction') || 'next';
                            console.log(`Image loaded: index=${index}, direction=${direction}`);
                            
                            // Apply smooth transition effect
                            window.applyImageTransition(direction);
                            
                            // Remove the trigger to avoid duplicate calls
                            trigger.removeAttribute('id');
                            trigger.setAttribute('data-processed', 'true');
                        }
                    });
                    
                    // Look for preloading triggers
                    const preloadTriggers = document.querySelectorAll('#preload-trigger');
                    preloadTriggers.forEach(trigger => {
                        const action = trigger.getAttribute('data-action');
                        console.log(`Found preload trigger with action: ${action}`);
                        
                        if (action === 'showPreloading') {
                            // Add preloading class to image container
                            const imageContainer = document.querySelector('.image-container');
                            if (imageContainer) {
                                imageContainer.classList.add('preloading');
                                console.log('Added preloading class to image container');
                            }
                            // Remove the trigger to avoid duplicate calls
                            trigger.removeAttribute('id');
                            trigger.setAttribute('data-processed', 'true');
                        } else if (action === 'hidePreloading') {
                            // Remove preloading class from image container
                            const imageContainer = document.querySelector('.image-container');
                            if (imageContainer) {
                                imageContainer.classList.remove('preloading');
                                console.log('Removed preloading class from image container');
                            }
                            // Remove the trigger to avoid duplicate calls
                            trigger.removeAttribute('id');
                            trigger.setAttribute('data-processed', 'true');
                        }
                    });
                    
                    // Always hide speed slider loading indicators
                    window.hideSpeedSliderLoading();
                }
            });
        });
        
        // Start observing the document
        observer.observe(document.body, { 
            childList: true, 
            subtree: true 
        });
        
        // Set up a specific observer for the speed slider
        const speedSliderObserver = new MutationObserver(function(mutations) {
            // Hide loading indicators whenever the speed slider changes
            window.hideSpeedSliderLoading();
        });
        
        // Find and observe the speed slider container
        setTimeout(() => {
            const speedSliderContainer = document.querySelector('.speed-slider-container');
            if (speedSliderContainer) {
                speedSliderObserver.observe(speedSliderContainer, { 
                    childList: true,
                    subtree: true,
                    attributes: true
                });
                console.log('Set up observer for speed slider container');
            }
        }, 1000);
    }

    // Image transition effects
    function setupImageTransitions() {
        // Create a container for preloaded images
        const preloadContainer = document.createElement('div');
        preloadContainer.style.display = 'none';
        preloadContainer.id = 'preload-container';
        document.body.appendChild(preloadContainer);
        
        // Function to preload an image
        window.preloadImage = function(src) {
            if (!src || window.imageCache[src]) return;
            
            const img = new Image();
            img.src = src;
            img.onload = function() {
                window.imageCache[src] = true;
                console.log(`Preloaded image: ${src}`);
            };
            preloadContainer.appendChild(img);
        };
        
        // Function to apply smooth transition between images
        window.applyImageTransition = function(direction) {
            // Find the current image viewer
            const container = document.querySelector('.custom-image-viewer');
            if (!container) return;
            
            // Get the current and next image elements
            const currentImg = container.querySelector('#current-image');
            const nextImg = container.querySelector('#next-image');
            
            if (!currentImg) return;
            
            // Apply transition effect based on direction
            if (direction === 'next' || direction === 'auto') {
                // Slide from right to left
                currentImg.style.transition = 'transform 0.2s ease-in-out, opacity 0.2s ease-in-out';
                currentImg.style.transform = 'translateX(0)';
                
                // Fade in effect
                currentImg.style.opacity = '0';
                setTimeout(() => {
                    currentImg.style.opacity = '1';
                    currentImg.style.transform = 'translateX(0)';
                }, 50);
            } else if (direction === 'prev') {
                // Slide from left to right
                currentImg.style.transition = 'transform 0.2s ease-in-out, opacity 0.2s ease-in-out';
                currentImg.style.transform = 'translateX(0)';
                
                // Fade in effect
                currentImg.style.opacity = '0';
                setTimeout(() => {
                    currentImg.style.opacity = '1';
                    currentImg.style.transform = 'translateX(0)';
                }, 50);
            } else {
                // Simple fade effect for other cases
                currentImg.style.transition = 'opacity 0.2s ease-in-out';
                currentImg.style.opacity = '0';
                setTimeout(() => {
                    currentImg.style.opacity = '1';
                }, 50);
            }
        };
        
        // Set up a mutation observer to watch for image changes
        const imageObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    // Look for new image viewers
                    const viewers = document.querySelectorAll('.custom-image-viewer');
                    viewers.forEach(viewer => {
                        if (!viewer.classList.contains('enhanced')) {
                            viewer.classList.add('enhanced');
                            console.log('Enhanced new image viewer');
                            
                            // Find images in this viewer
                            const images = viewer.querySelectorAll('img');
                            images.forEach(img => {
                                // Add transition properties
                                img.style.transition = 'opacity 0.2s ease-in-out, transform 0.2s ease-in-out';
                                img.style.willChange = 'opacity, transform';
                                img.style.backfaceVisibility = 'hidden';
                                img.style.transform = 'translateZ(0)';
                            });
                        }
                    });
                }
            });
        });
        
        // Start observing the document for image viewer changes
        imageObserver.observe(document.body, { 
            childList: true, 
            subtree: true 
        });
    }

    // Initialize all JavaScript functionality
    function initializeAll() {
        setupPlaybackControl();
        setupImageTransitions();
        console.log("Initialized all JavaScript functionality");
        
        // Add a global function to manually trigger playback (for debugging)
        window.manuallyStartPlayback = function(fps = 10) {
            console.log(`Manually starting playback at ${fps} FPS`);
            window.startPlayback(fps);
        };
        
        window.manuallyStopPlayback = function() {
            console.log('Manually stopping playback');
            window.stopPlayback();
        };
        
        // Run enhancement after a short delay to ensure all components are loaded
        setTimeout(() => {
            // Find and enhance all custom image viewers
            const viewers = document.querySelectorAll('.custom-image-viewer');
            console.log(`Found ${viewers.length} custom image viewers to enhance`);
            
            viewers.forEach(viewer => {
                viewer.classList.add('enhanced');
                
                // Find images in this viewer
                const images = viewer.querySelectorAll('img');
                images.forEach(img => {
                    // Add transition properties
                    img.style.transition = 'opacity 0.2s ease-in-out, transform 0.2s ease-in-out';
                    img.style.willChange = 'opacity, transform';
                    img.style.backfaceVisibility = 'hidden';
                    img.style.transform = 'translateZ(0)';
                });
            });
            
            // Hide all speed slider loading indicators
            if (typeof window.hideSpeedSliderLoading === 'function') {
                window.hideSpeedSliderLoading();
            }
            
            // Set up click handlers for the play/stop buttons
            setupPlaybackButtons();
        }, 1000);
        
        // Set up a periodic check to hide speed slider loading indicators
        setInterval(() => {
            if (typeof window.hideSpeedSliderLoading === 'function') {
                window.hideSpeedSliderLoading();
            }
        }, 500);
    }
    
    // Function to set up click handlers for the play/stop buttons
    function setupPlaybackButtons() {
        // Find the play button
        const playButtons = document.querySelectorAll('.controls-row button:nth-child(2)');
        if (playButtons.length > 0) {
            console.log(`Found ${playButtons.length} play buttons`);
            
            // Add a direct click handler to the play button
            playButtons.forEach(button => {
                // Remove any existing click handlers
                button.removeEventListener('click', window.playButtonClickHandler);
                
                // Add a new click handler
                window.playButtonClickHandler = function() {
                    console.log('Play button clicked directly');
                    
                    // Get the FPS value from the speed slider
                    const speedSlider = document.querySelector('.speed-slider-container input[type="range"]');
                    const fps = speedSlider ? parseFloat(speedSlider.value) : 10;
                    
                    // Start playback
                    window.startPlayback(fps);
                };
                
                button.addEventListener('click', window.playButtonClickHandler);
                console.log('Added direct click handler to play button');
            });
        }
        
        // Find the stop button
        const stopButtons = document.querySelectorAll('.controls-row button:nth-child(3)');
        if (stopButtons.length > 0) {
            console.log(`Found ${stopButtons.length} stop buttons`);
            
            // Add a direct click handler to the stop button
            stopButtons.forEach(button => {
                // Remove any existing click handlers
                button.removeEventListener('click', window.stopButtonClickHandler);
                
                // Add a new click handler
                window.stopButtonClickHandler = function() {
                    console.log('Stop button clicked directly');
                    window.stopPlayback();
                };
                
                button.addEventListener('click', window.stopButtonClickHandler);
                console.log('Added direct click handler to stop button');
            });
        }
    }

    // Call initialization
    initializeAll();
    
    // Set up a periodic check to reinitialize if needed
    setInterval(() => {
        // Check if the play button exists but doesn't have our click handler
        const playButtons = document.querySelectorAll('.controls-row button:nth-child(2)');
        if (playButtons.length > 0 && !window.playButtonClickHandler) {
            console.log('Reinitializing playback buttons');
            setupPlaybackButtons();
        }
    }, 2000);
    """

def get_css_for_image_viewer():
    """Get CSS for image viewer"""
    return """
    /* Custom image viewer */
    .custom-image-viewer {
        position: relative;
        width: 100%;
        height: 100%;
        overflow: hidden;
        background-color: #f5f5f5;
        border-radius: 8px;
    }
    
    .image-container {
        position: relative;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }
    
    .image-container img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        transition: opacity 0.2s ease-in-out, transform 0.2s ease-in-out;
        will-change: opacity, transform;
        backface-visibility: hidden;
        transform: translateZ(0);
    }
    
    #current-image {
        z-index: 2;
    }
    
    #next-image {
        position: absolute;
        z-index: 1;
        opacity: 0;
    }
    
    /* Controls styling */
    .controls-row {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    
    .player-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        background-color: #f9f9f9;
        margin-bottom: 15px;
    }
    
    .slider-container {
        padding: 10px 0;
    }
    
    /* Preload animation */
    @keyframes preload-pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .preloading {
        animation: preload-pulse 1s infinite;
    }
    
    /* Loading progress bar */
    .loading-progress-container {
        margin: 15px 0;
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f0f0;
    }
    
    .loading-progress-bar {
        height: 10px;
        background-color: #4CAF50;
        border-radius: 5px;
        transition: width 0.3s ease;
    }
    
    .loading-progress-text {
        text-align: center;
        margin-top: 5px;
        font-size: 0.9em;
        color: #555;
    }
    
    /* Memory usage indicator */
    .memory-usage {
        text-align: right;
        font-size: 0.8em;
        color: #777;
        margin-top: 5px;
    }
    
    /* Hide loading spinner for FPS slider - more specific selectors */
    .speed-slider-container .wrap .progress-container {
        display: none !important;
    }
    
    /* Additional selectors to hide loading indicators for the speed slider */
    .speed-slider-container .progress-container,
    .speed-slider-container .progress,
    .speed-slider-container .progress-bar,
    .speed-slider-container .wrap .progress,
    .speed-slider-container .wrap .progress-bar {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }
    
    /* Make sure the slider itself remains visible */
    .speed-slider-container input[type="range"] {
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Preload container */
    #preload-container {
        position: absolute;
        width: 1px;
        height: 1px;
        overflow: hidden;
        opacity: 0;
        pointer-events: none;
    }
    
    /* Image transition effects */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .fade-in {
        animation: fadeIn 0.2s ease-in-out;
    }
    
    .slide-in-right {
        animation: slideInRight 0.2s ease-in-out;
    }
    
    .slide-in-left {
        animation: slideInLeft 0.2s ease-in-out;
    }
    """

async def update_progress_callback(progress_component, progress_value, loaded, total, memory_mb=None):
    """
    Update progress bar and return HTML for progress indicator.
    
    Args:
        progress_component: Gradio Progress component
        progress_value: Progress value (0-1)
        loaded: Number of loaded images
        total: Total number of images
        memory_mb: Optional memory usage in MB
        
    Returns:
        HTML string for progress indicator
    """
    # Update the progress bar
    if progress_component:
        progress_component(progress_value, f"Loading images: {loaded}/{total}")
    
    # Create HTML for progress indicator
    progress_html = f"""
    <div class="loading-progress-container">
        <div class="loading-progress-bar" style="width: {progress_value * 100}%;"></div>
        <div class="loading-progress-text">
            Loaded {loaded} of {total} images ({progress_value * 100:.1f}%)
        </div>
    """
    
    # Add memory usage if provided
    if memory_mb is not None:
        progress_html += f"""
        <div class="memory-usage">
            Memory usage: {memory_mb:.1f} MB
        </div>
        """
    
    progress_html += "</div>"
    
    # Small delay to allow UI updates
    await asyncio.sleep(0.01)
    
    return progress_html 