from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time, os

caps = DesiredCapabilities().CHROME
caps["pageLoadStrategy"] = "none"
driver = webdriver.Chrome('/home/legmonkey/Documents/FFThesis/chromedriver', desired_capabilities=caps)
driver.get('https://www.pro-football-reference.com/boxscores')
time.sleep(1)

for i in range(2016, 2017):
    for j in range(7, 8):
        driver.find_element_by_xpath("//select[@name='year_id']/option[text()=" + str(i) + "]").click()
        time.sleep(1)
        driver.find_element_by_xpath("//select[@name='week']/option[text()='Week " + str(j) + "']").click()
        time.sleep(1)
        driver.find_elements_by_xpath("//input[@value='Find Games']")[1].click()
        time.sleep(1)
        f = driver.find_elements_by_xpath("//td[@class='right gamelink']/a")
        b = 0
        while b < len(f):
            temp = driver.find_elements_by_xpath("//td[@class='right gamelink']/a")
            k = temp[b]
            if k.text == "Final":
                k.click()
                time.sleep(1)
                a = None
                while a == None or name == None:
                    try:
                        time.sleep(1)
                        name = driver.find_element_by_xpath("//h1").text.split(' - ')[0]
                        ActionChains(driver).move_to_element(driver.find_element_by_xpath("//div[@id='all_player_offense']//li[@class='hasmore']")).perform()
                        time.sleep(1)
                        driver.find_element_by_xpath("//div[@id='all_player_offense']//button[@class='tooltip' and @tip='Export table as <br>suitable for use with excel']").click()
                        a = driver.find_element_by_xpath("//pre").text
                    except:
                        print ("stuck")
                        driver.refresh()
                filename = 'Data/gamebygame/' + str(i) + '/' + str(j) + '/' + name
                if not os.path.exists(os.path.dirname(filename)):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                h = open(filename, 'w')
                h.write(a)
                h.flush()
                h.close()
                driver.back()
                time.sleep(1)
            b += 1
        driver.back()
        time.sleep(1)
