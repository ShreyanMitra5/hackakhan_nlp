import serial


port = '/dev/tty.usbmodem11301'  
baudrate = 9600
ser = serial.Serial(port, baudrate, timeout=1)  

try:
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode().strip() 
            temp = float(data[0:5])
            humidity = float(data[6:])
            print(temp, humidity)
except KeyboardInterrupt:
    print("Program ended")
finally:
    ser.close()
