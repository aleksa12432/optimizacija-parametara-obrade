import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import tensorflow as tf

def add_intercept(x):
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x
    return new_x

def fit(X, y):
    return np.linalg.solve(np.dot(X.transpose(), X), np.dot(X.transpose(), y))

def htheta(theta, x_i):
    return float(np.dot(theta, x_i))

def j(theta, X, y):
    suma = 0
    for i in range(len(X)):
        suma += (htheta(theta, X[i]) - y[i])**2
    return (1 / len(X)) * suma

def polinomi(stepen, X):
    komb = []
    for i in range(2, stepen+1):
        for l in list(it.combinations_with_replacement(range(len(X[0])), i)):
            komb.append(l)

    print(komb)
    novoX = np.zeros((X.shape[0], X.shape[1] + len(komb)))
    novoX[:, 0:len(X[0])] = X
    
    for l in range(len(komb)):
        pomnozak = 1
        for i in komb[l]:
            pomnozak *= X[:, i]
        novoX[:, len(X[0]) + l] = pomnozak

    return novoX

csv = pd.read_csv("train.csv")
n = pd.DataFrame(csv, columns=["n"])
vf = pd.DataFrame(csv, columns=["Vf"])
ap = pd.DataFrame(csv, columns=["ap"])
y = pd.DataFrame(csv, columns=["Ra"])

zaSave = pd.DataFrame(csv, columns=["n", "Vf", "ap", "Ra"])
zaSave.rename(columns = {"Ra":"Ra_tacno"})


print("Чување графика n / Ra...")
plt.scatter(n.to_numpy(),y.to_numpy())
plt.xlabel("n")
plt.ylabel("Ra")
plt.savefig("n-Ra-grafik.png")
plt.close()

print("Чување графика Vf / Ra...")
plt.scatter(vf.to_numpy(),y.to_numpy())
plt.xlabel("Vf")
plt.ylabel("Ra")
plt.savefig("Vf-Ra-grafik.png")
plt.close()

print("Чување графика ap / Ra...")
plt.scatter(ap.to_numpy(),y.to_numpy())
plt.xlabel("ap")
plt.ylabel("Ra")
plt.savefig("ap-Ra-grafik.png")
plt.close()

print("====================================================")

print("Извршавам линеаран модел...") 
x_train = np.array(pd.DataFrame(csv, columns=["n","Vf","ap"]).to_numpy())
y_train = pd.DataFrame(csv, columns=["Ra"]).to_numpy().reshape(len(x_train),)
x_train = add_intercept(x_train)
theta = fit(x_train, y_train)

rezLinearan = np.zeros((len(x_train),1))

for i in range(len(x_train)):
    rezLinearan[i] = htheta(theta, x_train[i])

zaSave["Ra_linearan"] = rezLinearan

print("Тета резултат: ")
print(theta)
print("Средње квадратно одступање (J) модела: ", j(theta, x_train, y_train))
print(f"Функција модела: {theta[0]} + {theta[1]}*n + {theta[2]}*Vf + {theta[3]}*ap")
print("====================================================")
print("Нормализација параметара...")
x_train = pd.DataFrame(csv, columns=["n","Vf","ap"])

n_srednje = x_train["n"].mean()
Vf_srednje = x_train["Vf"].mean()
ap_srednje = x_train["ap"].mean()


n_opseg = x_train["n"].max() - x_train["n"].min()
Vf_opseg = x_train["Vf"].max() - x_train["Vf"].min()
ap_opseg = x_train["ap"].max() - x_train["ap"].min()

print(n_srednje, n_opseg)

x_train["n"] -= n_srednje
x_train["n"].div(n_opseg)

x_train["Vf"] -= Vf_srednje 
x_train["Vf"].div(Vf_opseg)

x_train["ap"] -= ap_srednje
x_train["ap"].div(ap_opseg)

x_train = add_intercept(x_train.to_numpy())

print("Извршавам линеаран модел над нормализованим скупом...")
theta = fit(x_train, y_train)
print("Тета резултат: ")
print(theta)
print("Средње квадратно одступање (J) модела: ", j(theta, x_train, y_train))
print(f"Функција модела: {theta[0]} + {theta[1]}*n + {theta[2]}*Vf + {theta[3]}*ap")
print(f"Где: n = n*{n_opseg} + {n_srednje}, Vf = Vf*{Vf_opseg} + {Vf_srednje}, ap = ap*{ap_opseg} + {ap_srednje}")

print("====================================================")
print("Креирам квадратне полиноме у скупу...")
x_train = np.array(pd.DataFrame(csv, columns=["n","Vf","ap"]).to_numpy())
X = add_intercept(polinomi(2, x_train))
print("Извршавам линеаран модел над скупом полинома...")
theta = fit(X, y_train)

rezLinearan2 = np.zeros((len(x_train),1))

for i in range(len(x_train)):
    rezLinearan2[i] = htheta(theta, X[i])

zaSave["Ra_kvadratni"] = rezLinearan2


print("Тета резултат: ")
print(theta)
print("Средње квадратно одступање (J) модела: ", j(theta, X, y_train))
print("====================================================")

print("Креирам кубне полиноме у скупу...")
x_train = np.array(pd.DataFrame(csv, columns=["n","Vf","ap"]).to_numpy())
X = add_intercept(polinomi(3, x_train))
print(X)
print("Извршавам линеаран модел над скупом полинома...")
theta = fit(X, y_train)
print("Тета резултат: ")
print(theta)
print("Средње квадратно одступање (J) модела: ", j(theta, X, y_train))

print("====================================================")
print("Чување графика полиномске регресије 3. степена...")
x_graph = np.arange(0, 64)
y_graph1 = np.zeros((64, 1))
for i in range(len(X)):
    y_graph1[i] = htheta(theta, X[i])

zaSave["Ra_kubni"] = y_graph1

y_graph2 = y_train
fig,ax = plt.subplots()
ax.plot(x_graph, y_graph1, marker="o", color="red", label="Предвиђена храпавост")
ax.set_xlabel("Тренинг пример")
ax.set_ylabel("Храпавост")
ax.plot(x_graph, y_graph2, marker="o", color="blue", label="Тачна храпавост")
ax.legend()
plt.savefig("linearna-regresija-3-stepen.png")
plt.close()
print("====================================================")
print("Тренирам неуронску мрежу са 5 скривених слојева...")
df = pd.DataFrame(csv, columns=["n","Vf","ap","Ra"])
train_df = df

max_val = train_df.max(axis= 0)
min_val = train_df.min(axis= 0)
 
range = max_val - min_val
train_df = (train_df - min_val)/(range)
X_train = train_df.drop('Ra',axis=1)
y_train = train_df['Ra']

input_shape = [X_train.shape[1]]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(units=1)])

model.summary()

model.compile(optimizer='adam', loss='mae') 

losses = model.fit(X_train, y_train, epochs=1000)

x_graph = np.arange(0, 64)
y_graph1 = model.predict(X_train)

y_graph1 = y_graph1 * range[3] + min_val[3]

zaSave["Ra_5_skrivenih_relu"] = y_graph1

y_graph2 = pd.DataFrame(csv, columns=["Ra"]).to_numpy().reshape(len(x_train),)
fig,ax = plt.subplots()
ax.plot(x_graph, y_graph1, marker="o", color="red", label="Предвиђена храпавост")
ax.set_xlabel("Тренинг пример")
ax.set_ylabel("Храпавост")
ax.plot(x_graph, y_graph2, marker="o", color="blue", label="Тачна храпавост")
ax.legend()
plt.savefig("neuronska-mreza-5-relu-hidden.png")

print("===================================================")
print("Тренирам неуронску мрежу са 7 скривених слојева...")
df = pd.DataFrame(csv, columns=["n","Vf","ap","Ra"])
train_df = df

max_val = train_df.max(axis= 0)
min_val = train_df.min(axis= 0)
 
range = max_val - min_val
train_df = (train_df - min_val)/(range)
X_train = train_df.drop('Ra',axis=1)
y_train = train_df['Ra']

input_shape = [X_train.shape[1]]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=7, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(units=1)])

model.summary()

model.compile(optimizer='adam', loss='mae') 

losses = model.fit(X_train, y_train, epochs=1000)

x_graph = np.arange(0, 64)
y_graph1 = model.predict(X_train)

y_graph1 = y_graph1 * range[3] + min_val[3]

zaSave["Ra_7_skrivenih_relu"] = y_graph1

y_graph2 = pd.DataFrame(csv, columns=["Ra"]).to_numpy().reshape(len(x_train),)
fig,ax = plt.subplots()
ax.plot(x_graph, y_graph1, marker="o", color="red", label="Предвиђена храпавост")
ax.set_xlabel("Тренинг пример")
ax.set_ylabel("Храпавост")
ax.plot(x_graph, y_graph2, marker="o", color="blue", label="Тачна храпавост")
ax.legend()
plt.savefig("neuronska-mreza-7-relu-hidden.png")

print("===================================================")
print("Тренирам неуронску мрежу са 10 скривених слојева...")
df = pd.DataFrame(csv, columns=["n","Vf","ap","Ra"])
train_df = df

max_val = train_df.max(axis= 0)
min_val = train_df.min(axis= 0)
 
range = max_val - min_val
train_df = (train_df - min_val)/(range)
X_train = train_df.drop('Ra',axis=1)
y_train = train_df['Ra']

input_shape = [X_train.shape[1]]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(units=1)])

model.summary()

model.compile(optimizer='adam', loss='mae') 

losses = model.fit(X_train, y_train, epochs=1000)

x_graph = np.arange(0, 64)
y_graph1 = model.predict(X_train)
y_graph1 = y_graph1 * range[3] + min_val[3]

zaSave["Ra_10_skrivenih_relu"] = y_graph1

y_graph2 = pd.DataFrame(csv, columns=["Ra"]).to_numpy().reshape(len(x_train),)
fig,ax = plt.subplots()
ax.plot(x_graph, y_graph1, marker="o", color="red", label="Предвиђена храпавост")
ax.set_xlabel("Тренинг пример")
ax.set_ylabel("Храпавост")
ax.plot(x_graph, y_graph2, marker="o", color="blue", label="Тачна храпавост")
ax.legend()
plt.savefig("neuronska-mreza-10-relu-hidden.png")

print("===================================================")
print("Тренирам неуронску мрежу са 5 скривених слојева (tanh)...")
df = pd.DataFrame(csv, columns=["n","Vf","ap","Ra"])
train_df = df

max_val = train_df.max(axis= 0)
min_val = train_df.min(axis= 0)
 
range = max_val - min_val
train_df = (train_df - min_val)/(range)
X_train = train_df.drop('Ra',axis=1)
y_train = train_df['Ra']

input_shape = [X_train.shape[1]]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5, activation='tanh', input_shape=input_shape),
    tf.keras.layers.Dense(units=1)])

model.summary()

model.compile(optimizer='adam', loss='mae') 

losses = model.fit(X_train, y_train, epochs=1000)

x_graph = np.arange(0, 64)
y_graph1 = model.predict(X_train)
y_graph1 = y_graph1 * range[3] + min_val[3]

zaSave["Ra_5_skrivenih_tanh"] = y_graph1

y_graph2 = pd.DataFrame(csv, columns=["Ra"]).to_numpy().reshape(len(x_train),)
fig,ax = plt.subplots()
ax.plot(x_graph, y_graph1, marker="o", color="red", label="Предвиђена храпавост")
ax.set_xlabel("Тренинг пример")
ax.set_ylabel("Храпавост")
ax.plot(x_graph, y_graph2, marker="o", color="blue", label="Тачна храпавост")
ax.legend()
plt.savefig("neuronska-mreza-5-tanh-hidden.png")

print("===================================================")
print("Тренирам неуронску мрежу са 2x5 скривених слојева...")
df = pd.DataFrame(csv, columns=["n","Vf","ap","Ra"])
train_df = df

max_val = train_df.max(axis= 0)
min_val = train_df.min(axis= 0)
 
range = max_val - min_val
train_df = (train_df - min_val)/(range)
X_train = train_df.drop('Ra',axis=1)
y_train = train_df['Ra']

input_shape = [X_train.shape[1]]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(units=5, activation='relu'),
    tf.keras.layers.Dense(units=1)])
 
model.summary()

model.compile(optimizer='adam', loss='mae') 

losses = model.fit(X_train, y_train, epochs=1000)

x_graph = np.arange(0, 64)
y_graph1 = model.predict(X_train)
y_graph1 = y_graph1 * range[3] + min_val[3]

zaSave["Ra_5x2_skrivenih_relu"] = y_graph1

y_graph2 = pd.DataFrame(csv, columns=["Ra"]).to_numpy().reshape(len(x_train),)
fig,ax = plt.subplots()
ax.plot(x_graph, y_graph1, marker="o", color="red", label="Предвиђена храпавост")
ax.set_xlabel("Тренинг пример")
ax.set_ylabel("Храпавост")
ax.plot(x_graph, y_graph2, marker="o", color="blue", label="Тачна храпавост")
ax.legend()
plt.savefig("neuronska-mreza-5x2-relu-hidden.png")

zaSave.to_csv("rezultati.csv")

print("===================================================")