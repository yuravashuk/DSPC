import math

pi = math.pi

# Base Shape class
class Shape:
    def __init__(self, name):
        self.name = name

    def area(self):
        pass

    def perimeter(self):
        pass

# Circle class
class Circle(Shape):
    def __init__(self, radius) -> None:
        super().__init__("Circle")
        self.radius = radius

    def area(self):
        return pi * self.radius ** 2

    def perimeter(self):
        return 2 * pi * self.radius
    

# Rectange class
class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
    
    def __str__(self) -> str:
        return f"{self.name} with width = {self.width} & height = {self.height}"
    

def main():
    circle = Circle(5)
    print(f"{circle.name} area: { circle.area() }, perimeter: { circle.perimeter() }")

    rectangle = Rectangle(5, 5)
    print(rectangle)

if __name__ == "__main__":
    main()