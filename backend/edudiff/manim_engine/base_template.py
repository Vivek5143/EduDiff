from manim import *

class BaseTemplate(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        
        # Default configuration for 1080p is standard in Manim, 
        # but we ensure white background here.
        
        # Placeholder for content generation
        self.generate_content()

    def generate_content(self):
        """
        Override this method in subclasses to add specific content.
        """
        pass
