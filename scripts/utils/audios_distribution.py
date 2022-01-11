import matplotlib.pyplot as plt


def distribution_audios(class_dist, text):
    print('----------------------')
    print('DISTRIBUCION DE AUDIOS')
    print('----------------------')

    fig, ax = plt.subplots()
    ax.set_title(text, y=1.08)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
           shadow=False, startangle=90)
    ax.axis('equal')
    plt.show()


def run(class_dist, text):
    distribution_audios(class_dist, text)


distribution_audios.run(class_dist, "Distribucion inicial de Audios")
