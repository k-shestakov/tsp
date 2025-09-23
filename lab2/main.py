import matplotlib.pyplot as plt

import random

def simulate_trains(num_trains=10000, mean_wagons=10, std_wagons=4, min_wagons=1):
	wagons_list = []
	for _ in range(num_trains):
		wagons = int(round(random.gauss(mean_wagons, std_wagons)))
		wagons = max(wagons, min_wagons)
		wagons_list.append(wagons)
	return wagons_list

def main():
	num_trains = 10000
	mean = 10
	std = 4
	wagons_list = simulate_trains(num_trains, mean, std)

	count_6_14 = sum(6 <= w <= 14 for w in wagons_list)
	percent_6_14 = count_6_14 / num_trains * 100
	print(f"Процент поездов с 6-14 вагонами: {percent_6_14:.2f}%")

	plt.hist(wagons_list, bins=range(1, 22), edgecolor='black', align='left', rwidth=0.8)
	plt.title('Распределение количества вагонов в поездах')
	plt.xlabel('Количество вагонов')
	plt.ylabel('Число поездов')
	plt.xticks(range(1, 21))
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	plt.show()

if __name__ == "__main__":
	main()
