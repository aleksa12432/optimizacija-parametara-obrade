# Оптимизација параметара обраде глодањем
#### Милан Маринковић 645/2019 и Алекса Лазаревић 601/2019

### Упутство:
1. Инсталирати python3 и pip3 и git на систему
  
  Arch/Manjaro:
  ```bash
  sudo pacman -S python python-pip git
  ```
  Debian/Ubuntu:
  ```bash
  sudo apt install python3 python3-pip git
  ```
  Fedora/CentOS
  ```bash
  sudo yum install python3 python3-pip git
  ```
2. Инсталирати неопходне библиотеке на pip3
  
  Arch/Manjaro:
  ```bash
  pip install matplotlib
  pip install numpy
  pip install pandas
  ```
  Остали:
  ```bash
  pip3 install matplotlib
  pip3 install numpy
  pip3 install pandas
  ```
3. Клонирати репозиторијум 
  ```bash
  git clone https://github.com/aleksa12432/optimizacija-parametara-obrade.git
  cd optimizacija-parametara-obrade
  ```
4. Покренути python скрипту
  
  Arch:
  ```bash
  python trening.py
  ```
  Остали:
  ```bash
  python3 trening.py
  ```
