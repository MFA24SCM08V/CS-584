import argparse
import csv
import numpy as np
from scipy import signal
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample


def generate_data(size=10000, noise=0.1, file_out="sample.csv"):
    rng = np.random.default_rng()
    sam_x = rng.uniform(-10, 10, size)
    noise = rng.normal(0, noise, size)
    offset = rng.uniform(1)
    sam_y = signal.sawtooth(2 * np.pi * sam_x + offset) + noise

    with open(file_out, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in zip(sam_x, sam_y):
            writer.writerow([x, y])


def fit_model(sam_x, sam_y, m, n):
    # Ensure sam_x is a 2D array with shape (n_samples, 1)
    sam_x = np.asarray(sam_x).reshape(-1, 1)
    sam_y = np.asarray(sam_y).reshape(-1, 1)
    

    ones = np.ones((sam_x.shape[0], 1))
    
   
    cos = np.hstack([np.cos(2 * np.pi * j * sam_x) for j in range(1, m + 1)])
    sin = np.hstack([np.sin(2 * np.pi * j * sam_x) for j in range(1, n + 1)])
    
   
    dmat = np.concatenate([ones, cos, sin], axis=1)
    

    coeffs, _, _, _ = np.linalg.lstsq(dmat, sam_y, rcond=None)
    
   
    pred = dmat @ coeffs
    rmse = np.sqrt(np.mean((sam_y - pred) ** 2))
    
   
    mod_str = model_to_string(coeffs.flatten(), m, n)
    
    return mod_str, rmse, coeffs

def model_to_string(coeffs, m, n):
    trem = [f"{coeffs[0]:.4f}"]  # Intercept
    trem += [f"{coeffs[j]:+.4f} cos(2π * {j})" for j in range(1, m + 1)]
    trem += [f"{coeffs[m + j]:+.4f} sin(2π * {j})" for j in range(1, n + 1)]
    return " + ".join(trem)


def calculate_aic(n, mse, param_numbers):
    aic = n * np.log(mse) + 2 * param_numbers
    return aic


def cross_validation(sam_x, sam_y, m, n, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=None)
    rmses = []
    
    for ind_train, ind_test in kf.split(sam_x):
        x_tr, x_tst = sam_x[ind_train], sam_x[ind_test]
        y_tr, y_tst = sam_y[ind_train], sam_y[ind_test]
        
        _, rmse, _ = fit_model(x_tr, y_tr, m, n)
        rmses.append(rmse)
        
    return np.mean(rmses)

def bootstrap(sam_x, sam_y, m, n, num_itr=1000):
    boot_rmses = []
    
    for _ in range(num_itr):
        x_bootstr, y_bootstr = resample(sam_x, sam_y)
        _, rmse, _ = fit_model(x_bootstr, y_bootstr, m, n)
        boot_rmses.append(rmse)
        
    return np.mean(boot_rmses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int, help="Number of cosine trem")
    parser.add_argument("n", type=int, help="Number of sine trem")
    parser.add_argument("-f", "--input_file", default="sample.csv", help="Name of input data file")
    args = parser.parse_args()

   
    data = []
    with open(args.input_file, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            data.append([float(line["x"]), float(line["y"])])
    
    data = np.array(data)
    sam_x = data[:, 0]
    sam_y = data[:, 1]

    
    mod_strified, rmse, coeffs = fit_model(sam_x, sam_y, args.m, args.n)
    print("Fitted model:", mod_strified)
    print(f"RMSE: {rmse}")

    
    param_numbers = args.m + args.n + 1  # +1 for the intercept
    mse = rmse**2
    aic = calculate_aic(len(sam_x), mse, param_numbers)
    print(f"AIC: {aic}")


    cv_rmse_5 = cross_validation(sam_x, sam_y, args.m, args.n, 5)
    cv_rmse_10 = cross_validation(sam_x, sam_y, args.m, args.n, 10)
    print(f"5-Fold CV RMSE: {cv_rmse_5}")
    print(f"10-Fold CV RMSE: {cv_rmse_10}")


    bootstrap1 = bootstrap(sam_x, sam_y, args.m, args.n)
    print(f"Bootstrap Mean RMSE: {bootstrap1}")
   

if __name__ == "__main__":
    main()
