class   Thresholds:
    def __init__(self):
        self.draw_red = 255
        self.blur_1 = 3
        self.blur_sigma = 1
        self.canny_1 = 255
        self.canny_2 = 0
        self.normalize_alpha = 0
        self.normalize_beta = 255
        self.invert = True
        self.hoare_dp = 1.2
        self.hoare_min_dist = 0
        self.hoare_param1 = 2
        self.hoare_param2 = 2
        self.hoare_min_radius = 1
        self.hoare_max_radius = 1
        self.min_ellipse = 10
        self.max_ellipse = 50
        self.min_ellipse_aspect = 0.4
        self.max_ellipse_aspect = 1.0
        self.draw_original = False
