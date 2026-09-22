"""Status components."""

import json
import customtkinter as ctk

__all__ = ['ProgressDialog', 'StatusBar']


class ProgressDialog(ctk.CTkToplevel):
    """
    Modal dialog showing progress bar and status message.
    """
    def __init__(self, parent, title="Processing"):
        super().__init__(parent)
        self.title(title)
        self.geometry("500x150")
        self.resizable(False, False)
        
        self.grab_set()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (500 // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (150 // 2)
        self.geometry(f"+{x}+{y}")
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="Initializing...",
            font=("Arial", 14)
        )
        self.status_label.pack(pady=(20, 10))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self, width=400)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        # Percentage label
        self.percent_label = ctk.CTkLabel(
            self,
            text="0%",
            font=("Arial", 12)
        )
        self.percent_label.pack(pady=5)
        
        # Cancel button
        self.cancelled = False
        self.cancel_button = ctk.CTkButton(
            self,
            text="Cancel",
            command=self.cancel,
            width=100
        )
        self.cancel_button.pack(pady=10)
        
        self.protocol("WM_DELETE_WINDOW", self.cancel)
    
    def update_progress(self, fraction: float, message: str = ""):
        """Update progress bar and status message."""
        self.progress_bar.set(fraction)
        self.percent_label.configure(text=f"{int(fraction * 100)}%")
        if message:
            self.status_label.configure(text=message)
        self.update()
    
    def cancel(self):
        """Mark as cancelled and close."""
        self.cancelled = True
        self.destroy()
    
    def is_cancelled(self) -> bool:
        """Check if user cancelled the operation."""
        return self.cancelled


class StatusBar(ctk.CTkFrame):
    """
    Status bar to show information at the bottom of the window.
    Supports animated spinners for processing states.
    """
    def __init__(self, master, spinner_name: str = "dots", **kwargs):
        super().__init__(master, height=30, **kwargs)
        
        self.label = ctk.CTkLabel(
            self,
            text="Ready",
            anchor="w"
        )
        self.label.pack(side="left", padx=10, fill="x", expand=True)
        
        # Spinner state
        self.spinner_active = False
        self.spinner_frames = []
        self.spinner_interval = 80
        self.spinner_index = 0
        self.spinner_message = ""
        self.spinner_after_id = None
        
        # Load spinner configuration
        self._load_spinner(spinner_name)
    
    def _load_spinner(self, spinner_name: str):
        """Load spinner configuration from spinners.json."""
        try:
            import json
            from pathlib import Path
            
            # Look for spinners.json in the application root
            spinner_file = Path(__file__).resolve().parent.parent / "spinners.json"
            if spinner_file.exists():
                with open(spinner_file, 'r', encoding='utf-8') as f:
                    spinners = json.load(f)
                    if spinner_name in spinners:
                        spinner_config = spinners[spinner_name]
                        self.spinner_frames = spinner_config.get('frames', ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
                        self.spinner_interval = spinner_config.get('interval', 80)
                    else:
                        # Default fallback
                        self.spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                        self.spinner_interval = 80
            else:
                # Default fallback if file not found
                self.spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                self.spinner_interval = 80
        except Exception as e:
            print(f"Error loading spinner: {e}")
            # Simple fallback
            self.spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            self.spinner_interval = 80
    
    def set_status(self, message: str, spinning: bool = False):
        """
        Update status message.
        
        Args:
            message: The status message to display
            spinning: If True, starts animated spinner. If False, stops any active spinner.
        """
        if spinning:
            self.start_spinner(message)
        else:
            self.stop_spinner()
            self.label.configure(text=message)
            self.update()
    
    def start_spinner(self, message: str):
        """Start the animated spinner with the given message."""
        self.spinner_message = message
        self.spinner_active = True
        self.spinner_index = 0
        self._animate_spinner()
    
    def stop_spinner(self):
        """Stop the animated spinner."""
        self.spinner_active = False
        if self.spinner_after_id is not None:
            self.after_cancel(self.spinner_after_id)
            self.spinner_after_id = None
    
    def _animate_spinner(self):
        """Internal method to animate the spinner."""
        if not self.spinner_active:
            return
        
        frame = self.spinner_frames[self.spinner_index]
        self.label.configure(text=f"{frame} {self.spinner_message}")
        self.update()
        
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        self.spinner_after_id = self.after(self.spinner_interval, self._animate_spinner)
