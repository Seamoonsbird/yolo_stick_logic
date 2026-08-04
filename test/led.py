#!/usr/bin/env python
import RPi.GPIO as GPIO
import time

led_pin = 12 


def LED():
	
	
	GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
	GPIO.setup(led_pin, GPIO.OUT)
	GPIO.output(led_pin, GPIO.LOW)
	print("Starting demo now! Press CTRL+C to exit")

	while True:
		GPIO.output(led_pin, GPIO.HIGH)
		time.sleep(2)
		GPIO.output(led_pin, GPIO.LOW)
		time.sleep(2)


if __name__ == '__main__':
	LED()