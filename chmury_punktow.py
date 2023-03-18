from scipy.stats import norm
from csv import writer
import numpy as np


def generate_points_horizontal(num_points: int = 5000):
    distribution_x = norm(loc=100, scale=10)
    distribution_y = norm(loc=0, scale=20)
    distribution_z = norm(loc=0.2, scale=0.05)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points


def generate_points_vertical(num_points: int = 5000):
    distribution_x = norm(loc=50, scale=0.05)
    distribution_y = norm(loc=0, scale=20)
    distribution_z = norm(loc=0.2, scale=10)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points

def generate_points_cylinder(radius=10, height=50, density=5000, turns=400):
    angle = np.linspace(0, turns * np.pi, density)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = np.linspace(0, height, num=density)

    points = zip(x, y, z)
    return points

if __name__ == '__main__':
    cloud_points_horizontal = generate_points_horizontal(5000)
    cloud_points_vertical = generate_points_vertical(5000)
    cloud_points_cylinder = generate_points_cylinder()
    with open('LidarData.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)

        for p in cloud_points_horizontal:
            csvwriter.writerow(p)

        for p in cloud_points_vertical:
            csvwriter.writerow(p)

        for p in cloud_points_cylinder:
            csvwriter.writerow(p)