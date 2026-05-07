
import pygame

pygame.mixer.init()
pygame.mixer.music.load("mixkit-sound-alert-in-hall-1006.wav")

def play_alert():
    # tránh phát chồng âm thanh
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()