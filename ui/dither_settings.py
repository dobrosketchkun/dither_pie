"""Dither settings components."""

import tkinter as tk
import customtkinter as ctk

__all__ = ['DitherSettingsDialog']


class DitherSettingsDialog(ctk.CTkToplevel):
    """
    Generic settings dialog for dithering mode parameters.
    Automatically builds UI based on parameter metadata.
    """
    def __init__(self, parent, mode_name: str, parameter_info: dict,
                 current_values: dict = None, on_change=None, on_cancel=None, on_apply=None):
        super().__init__(parent)
        self.title(f"{mode_name} Settings")
        self.geometry("450x600")
        self.resizable(False, True)
        
        self.transient(parent)
        self.grab_set()
        self.lift()
        self.focus_force()
        
        self.parameter_info = parameter_info
        self.current_values = current_values or {}
        self.result_values = None
        self.widgets = {}
        self.on_change = on_change
        self.on_cancel_callback = on_cancel
        self.on_apply_callback = on_apply
        self._live_change_job = None
        
        # Title
        title = ctk.CTkLabel(
            self,
            text=f"Configure {mode_name}",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(15, 10))
        
        # Scrollable frame for parameters
        self.scroll_frame = ctk.CTkScrollableFrame(self, height=400)
        self.scroll_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Build parameter controls
        self._build_parameter_widgets()
        
        # Buttons
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=15, padx=15, fill='x')
        
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self.on_cancel,
            width=120
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="Apply",
            command=self.on_apply,
            width=120
        ).pack(side="right", padx=5)
        
        # Reset button in the middle
        ctk.CTkButton(
            btn_frame,
            text="Reset to Defaults",
            command=self.on_reset,
            width=140
        ).pack(side="left", padx=20, expand=True)
    
    def _build_parameter_widgets(self):
        """Build UI widgets for each parameter."""
        row = 0
        
        for param_name, param_info in self.parameter_info.items():
            # Label
            label_text = param_info.get('label', param_name)
            label = ctk.CTkLabel(
                self.scroll_frame,
                text=label_text + ":",
                font=("Arial", 12, "bold")
            )
            label.grid(row=row, column=0, sticky='w', padx=10, pady=(10, 2))
            row += 1
            
            # Description
            desc = param_info.get('description', '')
            if desc:
                desc_label = ctk.CTkLabel(
                    self.scroll_frame,
                    text=desc,
                    font=("Arial", 10),
                    text_color="gray"
                )
                desc_label.grid(row=row, column=0, sticky='w', padx=10, pady=(0, 5))
                row += 1
            
            # Widget based on type
            param_type = param_info.get('type')
            current_value = self.current_values.get(param_name, param_info.get('default'))
            
            if param_type == 'int':
                widget = self._create_int_widget(param_name, param_info, current_value)
            elif param_type == 'float':
                widget = self._create_float_widget(param_name, param_info, current_value)
            elif param_type == 'choice':
                widget = self._create_choice_widget(param_name, param_info, current_value)
            else:
                widget = None
            
            if widget:
                widget.grid(row=row, column=0, sticky='ew', padx=10, pady=(0, 5))
                self.widgets[param_name] = widget
                row += 1
        
        # Configure column weight
        self.scroll_frame.grid_columnconfigure(0, weight=1)
    
    def _create_int_widget(self, param_name, param_info, current_value):
        """Create widget for integer parameter."""
        frame = ctk.CTkFrame(self.scroll_frame)
        
        min_val = param_info.get('min', 0)
        max_val = param_info.get('max', 100)
        
        # Entry field
        entry = ctk.CTkEntry(frame, width=80)
        entry.insert(0, str(int(current_value)))
        entry.bind("<KeyRelease>", lambda _e: self._schedule_live_change())
        entry.bind("<FocusOut>", lambda _e: self._schedule_live_change())
        entry.pack(side='left', padx=(0, 10))
        
        # Range label
        range_label = ctk.CTkLabel(
            frame,
            text=f"({min_val} - {max_val})",
            font=("Arial", 10),
            text_color="gray"
        )
        range_label.pack(side='left')
        
        frame.entry = entry
        frame.min_val = min_val
        frame.max_val = max_val
        
        return frame
    
    def _create_float_widget(self, param_name, param_info, current_value):
        """Create widget for float parameter."""
        frame = ctk.CTkFrame(self.scroll_frame)
        
        min_val = param_info.get('min', 0.0)
        max_val = param_info.get('max', 1.0)
        step = param_info.get('step', 0.1)
        
        # Entry field
        entry = ctk.CTkEntry(frame, width=80)
        entry.insert(0, f"{current_value:.2f}")
        entry.bind("<KeyRelease>", lambda _e: self._schedule_live_change())
        entry.bind("<FocusOut>", lambda _e: self._schedule_live_change())
        entry.pack(side='left', padx=(0, 10))
        
        # Range label
        range_label = ctk.CTkLabel(
            frame,
            text=f"({min_val} - {max_val}, step {step})",
            font=("Arial", 10),
            text_color="gray"
        )
        range_label.pack(side='left')
        
        frame.entry = entry
        frame.min_val = min_val
        frame.max_val = max_val
        frame.step = step
        
        return frame
    
    def _create_choice_widget(self, param_name, param_info, current_value):
        """Create widget for choice parameter."""
        choices = param_info.get('choices', [])
        
        var = tk.StringVar(value=current_value)
        menu = ctk.CTkOptionMenu(
            self.scroll_frame,
            variable=var,
            values=choices,
            command=lambda _v: self._schedule_live_change()
        )
        menu.var = var
        
        return menu
    
    def _get_widget_value(self, param_name):
        """Extract value from widget with validation."""
        widget = self.widgets.get(param_name)
        if not widget:
            return None
        
        param_info = self.parameter_info[param_name]
        param_type = param_info.get('type')
        
        if param_type == 'int':
            try:
                value = int(widget.entry.get())
                # Clamp to valid range
                value = max(widget.min_val, min(widget.max_val, value))
                return value
            except ValueError:
                # Return default if invalid
                return param_info.get('default', widget.min_val)
        elif param_type == 'float':
            try:
                value = float(widget.entry.get())
                # Clamp to valid range
                value = max(widget.min_val, min(widget.max_val, value))
                return value
            except ValueError:
                # Return default if invalid
                return param_info.get('default', widget.min_val)
        elif param_type == 'choice':
            return widget.var.get()
        
        return None
    
    def on_reset(self):
        """Reset all parameters to defaults."""
        for param_name, widget in self.widgets.items():
            param_info = self.parameter_info[param_name]
            default_value = param_info.get('default')
            param_type = param_info.get('type')
            
            if param_type == 'int':
                widget.entry.delete(0, tk.END)
                widget.entry.insert(0, str(int(default_value)))
            elif param_type == 'float':
                widget.entry.delete(0, tk.END)
                widget.entry.insert(0, f"{default_value:.2f}")
            elif param_type == 'choice':
                widget.var.set(default_value)
        self._schedule_live_change()

    def _schedule_live_change(self):
        """Debounce live change notifications."""
        if not self.on_change:
            return
        if self._live_change_job is not None:
            self.after_cancel(self._live_change_job)
        self._live_change_job = self.after(250, self._emit_live_change)

    def _emit_live_change(self):
        """Emit current parameter values for live preview."""
        self._live_change_job = None
        if not self.on_change:
            return
        current_values = {}
        for param_name in self.parameter_info.keys():
            current_values[param_name] = self._get_widget_value(param_name)
        self.on_change(current_values)
    
    def on_apply(self):
        """Apply settings and close."""
        self.result_values = {}
        for param_name in self.parameter_info.keys():
            self.result_values[param_name] = self._get_widget_value(param_name)
        if self.on_apply_callback:
            self.on_apply_callback(self.result_values)
        self.destroy()
    
    def on_cancel(self):
        """Cancel and close."""
        self.result_values = None
        if self.on_cancel_callback:
            self.on_cancel_callback()
        self.destroy()
