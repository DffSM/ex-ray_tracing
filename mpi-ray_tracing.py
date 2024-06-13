from mpi4py import MPI
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def reflect(vector, normal):
    return vector - 2 * np.dot(vector, normal) * normal

def intersect_ray(ray_origin, ray_direction, objects):
    distances = [np.linalg.norm(ray_direction) + np.random.random() for _ in objects]
    nearest_object = np.argmin(distances)
    return nearest_object, distances[nearest_object]

def phong_lighting(point, normal, view_direction, lights, object_color):
    ambient = 0.1
    color = np.zeros(3) + ambient * object_color
    for light in lights:
        light_dir = light['position'] - point
        light_dir /= np.linalg.norm(light_dir)
        diffuse = max(np.dot(normal, light_dir), 0)
        specular = np.dot(normal, reflect(-light_dir, normal)) ** 32
        color += light['color'] * (diffuse + specular)
    return color

def trace_ray(scene, lights, origin, direction, depth=0):
    if depth > 3:
        return np.zeros(3)
    color = np.zeros(3)
    objects = scene['objects']
    nearest_object, distance = intersect_ray(origin, direction, objects)
    if distance < float('inf'):
        point = origin + direction * distance
        normal = np.random.randn(3)
        normal /= np.linalg.norm(normal)
        color += phong_lighting(point, normal, -direction, lights, objects[nearest_object]['color'])
    return color

def worker(y_range, width, height, samples, scene, lights):
    result = np.zeros((len(y_range), width, 3), dtype=np.float32)
    for i, y in enumerate(y_range):
        for x in range(width):
            pixel_color = np.zeros(3)
            for _ in range(samples):
                direction = np.array([x - width / 2, y - height / 2, height])
                direction = direction / np.linalg.norm(direction)
                pixel_color += trace_ray(scene, lights, np.array([0, 0, 0]), direction)
            result[i, x, :] = pixel_color / samples
    return result

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    width, height = 800, 600
    samples = 100
    scene = {'objects': [{'color': np.array([1, 1, 1])}]}

    lights = [
        {'position': np.array([300, 300, 100]), 'color': np.array([1, 0, 0]), 'intensity': 1.0},
        {'position': np.array([-300, -300, 100]), 'color': np.array([0, 0, 1]), 'intensity': 1.0}
    ]

    if rank == 0:
        print("Starting ray tracing...")
        start_time = time.time()

    y_range = np.array_split(range(height), size)[rank]

    local_result = worker(y_range, width, height, samples, scene, lights)

    result = None
    if rank == 0:
        result = np.zeros((height, width, 3), dtype=np.float32)

    comm.Gather(local_result, result, root=0)

    if rank == 0:
        end_time = time.time()
        print(f"Ray tracing completed in {end_time - start_time:.2f} seconds.")
        plt.imshow(result.clip(0, 1))
        plt.title("Advanced Ray Tracing Output")
        plt.show()

if __name__ == "__main__":
    main()
