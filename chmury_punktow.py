from scipy.stats import norm
from csv import writer


def generate_points_z(num_points: int = 2000):
    distribution_x = norm(loc=0, scale=0.05)
    distribution_y = norm(loc=0, scale=20)
    distribution_z = norm(loc=0.2, scale=200)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points


def generate_points_x(num_points: int = 2000):
    distribution_x = norm(loc=0, scale=200)
    distribution_y = norm(loc=0, scale=20)
    distribution_z = norm(loc=0.2, scale=0.05)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z)
    return points


if __name__ == '__main__':
    cloud_points_x = generate_points_x(2000)
    cloud_points_z = generate_points_z(2000)
    with open('LidarData.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
        csvwriter = writer(csvfile)

        for p in cloud_points_x:
            csvwriter.writerow(p)

        for p in cloud_points_z:
            csvwriter.writerow(p)
