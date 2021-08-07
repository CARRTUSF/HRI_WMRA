import sys
import pygame


class Button:
    def __init__(self, color, width, height, text=''):
        self.color = color
        self.width = width
        self.height = height
        self.text = text

    def draw(self, win, x, y, outline=None):
        # Call this method to draw the button on the screen
        if outline:
            pygame.draw.rect(win, outline, (x - 2, y - 2, self.width + 4, self.height + 4), 0)

        pygame.draw.rect(win, self.color, (x, y, self.width, self.height), 0)

        if self.text != '':
            font = pygame.font.SysFont('comicsans', 30)
            text = font.render(self.text, True, (0, 0, 0))
            win.blit(text, (
                x + (self.width / 2 - text.get_width() / 2), y + (self.height / 2 - text.get_height() / 2)))

    def is_mouse_over(self, pos, x, y):
        # Pos is the mouse position or a tuple of (x,y) coordinates
        if x < pos[0] < x + self.width:
            if y < pos[1] < y + self.height:
                return True
        return False


def generate_button_position(ref_x, ref_y, dx, dy, button_h, win):
    w, h = win.get_size()
    p_y = ref_y + dy
    if p_y + button_h > h - 4:
        p_y = ref_y - dy - button_h
    if ref_x + dx > w - 4:
        p_x = w - 4 - 2 * dx
    else:
        p_x = max(4, ref_x - dx)
    return p_x, p_y


def user_input_confirm_or_cancel_gui(win, button_w, button_h, ref_x, ref_y, dy, button_gap,
                                     confirm_color=(255, 255, 0), cancel_color=(0, 255, 255)):
    confirm_button = Button(confirm_color, button_w, button_h, 'Confirm')
    confirm_button_px, confirm_button_py = generate_button_position(ref_x, ref_y,
                                                                    button_w + button_gap / 2, dy,
                                                                    button_h, win)
    cancel_button = Button(cancel_color, button_w, button_h, 'Cancel')
    cancel_button_px = confirm_button_px + button_w + button_gap
    confirm_button.draw(win, confirm_button_px, confirm_button_py)
    cancel_button.draw(win, cancel_button_px, confirm_button_py)
    pygame.display.update()
    user_confirmed = False
    user_cancelled = False
    while not (user_confirmed or user_cancelled):
        for confirmation_eve in pygame.event.get():
            if confirmation_eve.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if confirmation_eve.type == pygame.MOUSEBUTTONDOWN:
                if confirmation_eve.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    if confirm_button.is_mouse_over((mx, my), confirm_button_px, confirm_button_py):
                        user_confirmed = True
                        print('User confirmed!')
                    elif cancel_button.is_mouse_over((mx, my), cancel_button_px, confirm_button_py):
                        user_cancelled = True
                        print('User cancelled!')
                    else:
                        pass
    return user_confirmed
