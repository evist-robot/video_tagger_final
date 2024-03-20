import asyncio
import websockets
import cv2
import base64
import datetime
import numpy as np
from ultralytics import YOLO
import json
import uuid

class ClientHandler:
    def __init__(self):
        # Ваши переменные и инициализации для каждого клиента
        self.model = YOLO("nano_4700.pt")
        # Добавьте другие переменные и инициализации, если необходимо
        self.no_helmet_frame_counter = 0
        self.no_uniform_frame_counter = 0
        self.no_googles_frame_counter = 0
        self.no_gloves_frame_counter = 0

        self.no_helmet_counter = 0
        self.no_uniform_counter = 0
        self.no_googles_counter = 0
        self.no_gloves_counter = 0

        self.no_helmet_detected = False
        self.no_uniform_detected = False
        self.no_googles_detected = False
        self.no_gloves_detected = False

        self.out = None
        self.non_detected_counter = 0
        self.frame_len = 0
        self.threshold = 0.5

        self.last_activity_time = datetime.datetime.now()

    async def send_json_data(self, websocket, data):
        json_data = json.dumps(data)
        await websocket.send(json_data)

    async def detect_objects(self, frame, formatted_now):

        class_ids = []
        class_ids_sorted = []

        scores_helmet = []
        scores_uniform = []
        scores_googles = []
        scores_gloves = []

        # при начале обработки кадра изначально alert False
        alert_detected = False

        json_list = []
        alert_begin_json = None
        alert_stop_json = None
        alert_count = None

        results = self.model(frame)[0]
        boxes = results.boxes.xyxy.cpu()
        clss = results.boxes.cls.cpu().tolist()

        # получаем список обнаруженных при детекции объектов
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            print('Обнаружены следующие объекты', class_ids)

        if 1.0 not in class_ids:
            if self.no_helmet_frame_counter > 0:
                self.no_helmet_frame_counter -= 1

        if 3.0 not in class_ids:
            if self.no_uniform_frame_counter > 0:
                self.no_uniform_frame_counter -= 1

        if 5.0 not in class_ids:
            if self.no_googles_frame_counter > 0:
                self.no_googles_frame_counter -= 1

        if 7.0 not in class_ids:
            if self.no_gloves_frame_counter > 0:
                self.no_gloves_frame_counter -= 1

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            print(f'Класс: {class_id}, счет {score}')
            # условия по уверенности модели в той или иной детекции (если уверенность ниже threshold, снимается балл уверенности)
            # в случае если одинаковых объектов на кадре много, снимается все равно только один балл (отсчет по max)
            if class_id == 1:
                scores_helmet.append(score)
                if max(scores_helmet) < self.threshold:  # если максимальный счет меньше 0.5 !!
                    if self.no_helmet_frame_counter > 0:
                        self.no_helmet_frame_counter -= 1

            if class_id == 3:
                scores_uniform.append(score)
                if max(scores_uniform) < self.threshold:  # если максимальный счет меньше 0.5 !!
                    if self.no_uniform_frame_counter > 0:
                        self.no_uniform_frame_counter -= 1

            if class_id == 5:
                scores_googles.append(score)
                if max(scores_googles) < self.threshold:  # если максимальный счет меньше 0.5 !!
                    if self.no_googles_frame_counter > 0:
                        self.no_googles_frame_counter -= 1

            if class_id == 7:
                scores_gloves.append(score)
                if max(scores_gloves) < self.threshold:  # если максимальный счет меньше 0.5 !!
                    if self.no_gloves_frame_counter > 0:
                        self.no_gloves_frame_counter -= 1

            if (class_id == 1.0 or class_id == 3.0 or class_id == 5.0 or class_id == 7.0) and score > self.threshold:
                now = datetime.datetime.now()
                formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")
                class_ids_sorted.append(class_id)

                # разбор объекта 1, остальные по логике однотипны
                # если в списке есть объект 1, но общее количество баллов уверенности менее 10, то добавляем балл
                if class_id == 1.0:
                    if class_ids_sorted.count(1.0) == 1 and self.no_helmet_frame_counter < 10:
                        self.no_helmet_frame_counter += 1
                        print('no_helmet_frame_counter', self.no_helmet_frame_counter)
                    # если баллы уверенности достигают 10, начинается отрисовка и идет alert
                    if self.no_helmet_frame_counter >= 10:
                        text = 'NO HELMET'
                        tracked_no_helmet = {
                            "type": "object",
                            "time": formatted_now,
                            "rect": [x1, x2, y1, y2],
                            "class": text
                        }

                        # устанавливаем состояния
                        alert_detected = True
                        self.no_helmet_detected = True

                        # здесь используем счетчик обнаружений
                        # если количество 1 в списке детекции больше текущего количества (на начоло 0), то разница прибавляется
                        if class_ids_sorted.count(1.0) > self.no_helmet_counter:
                            self.no_helmet_counter += (class_ids_sorted.count(1.0) - self.no_helmet_counter)

                        json_list.append(tracked_no_helmet)

                # то же с униформой
                if class_id == 3.0:
                    if class_ids_sorted.count(3.0) == 1 and self.no_uniform_frame_counter < 10:
                        self.no_uniform_frame_counter += 1
                    if self.no_uniform_frame_counter >= 10:
                        text = 'NO UNIFORM'
                        tracked_no_uniform = {
                            "type": "object",
                            "time": formatted_now,
                            "rect": [x1, x2, y1, y2],
                            "class": text
                        }

                        alert_detected = True
                        self.no_uniform_detected = True

                        if class_ids_sorted.count(3.0) > self.no_uniform_counter:
                            self.no_uniform_counter += (class_ids_sorted.count(3.0) - self.no_uniform_counter)

                        json_list.append(tracked_no_uniform)

                if class_id == 5.0:
                    if class_ids_sorted.count(5.0) == 1 and self.no_googles_frame_counter < 10:
                        self.no_googles_frame_counter += 1
                    if self.no_googles_frame_counter >= 10:
                        text = 'NO GOGGLES'
                        tracked_no_goggles = {
                            "type": "object",
                            "time": formatted_now,
                            "rect": [x1, x2, y1, y2],
                            "class": text
                        }

                        alert_detected = True
                        self.no_googles_detected = True

                        if class_ids_sorted.count(5.0) > self.no_googles_counter:
                            self.no_googles_counter += (class_ids_sorted.count(5.0) - self.no_googles_counter)

                        json_list.append(tracked_no_goggles)

                if class_id == 7.0:
                    if class_ids_sorted.count(7.0) == 1 and self.no_gloves_frame_counter < 10:
                        self.no_gloves_frame_counter += 1
                    if self.no_gloves_frame_counter >= 10:
                        text = 'NO GLOVES'
                        tracked_no_gloves = {
                            "type": "object",
                            "time": formatted_now,
                            "rect": [x1, x2, y1, y2],
                            "class": text
                        }

                        alert_detected = True
                        self.no_gloves_detected = True

                        if class_ids_sorted.count(7.0) > self.no_gloves_counter:
                            self.no_gloves_counter += (class_ids_sorted.count(7.0) - self.no_gloves_counter)

                        json_list.append(tracked_no_gloves)

        if alert_detected is True:
            print('Падаю в условие что alert_detected is True')
            self.non_detected_counter = 0
            if self.out is None:  # если видеозапись не начата - начинаем ее
                print('Падаю в условие что self.out is None')

                # now = datetime.datetime.now()
                # formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")
                # current_recording_name = f'Alert_{formatted_now}.mp4'
                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # out = cv2.VideoWriter(current_recording_name, fourcc, 8.0,
                #                       (im0.shape[1], im0.shape[0]))
                self.out = 'someword'
                # при начале записи видео отправляется первоначальный alert
                # здесь ситуация дана общая как unsafe_condition, поэтому объединили
                # если ситуацию описываем каждую отдельно, разкомментировать и переписать первую на no_helmet
                if self.no_helmet_detected == True or self.no_uniform_detected == True or self.no_googles_detected == True or self.no_gloves_detected == True:
                    global id
                    id = str(uuid.uuid4())
                    alert_begin_json = {
                        "type": "alert",
                        "id": id,
                        "situation": "unsafe_condition",
                        "detected": True,
                        "time": formatted_now
                    }
                    print('Начало алерта', alert_begin_json)

                    # await send_json_data(websocket, alert_begin_json)
        #

        try:
            # если alert нигде не стал True (изначально он False при обработке кадра, True становится при 10 баллах уверенности)
            if alert_detected is False:
                # начинаем отсчет фрагментов до выключения записи
                self.non_detected_counter += 1
                print('отсчет', self.non_detected_counter)
                # out.write(im0)
                # если насчитывает 50 кадров без детекции и при этом еще работает видеозапись - запись отключается
                if self.non_detected_counter >= 15:
                    if self.out is not None:
                        # out.release()
                        self.out = None

                        # alert останавливается, посылается соответствующий сигнал, а также общий подсчет объектов в этой записи (максимальное количество в кадре единовременно)
                        alert_stop_json = {
                            "type": "no_alert",
                            "id": id,
                            "situation": "no_unsafe_condition",
                            "detected": False,
                            "time": formatted_now
                        }
                        print('Конец алерта', alert_stop_json)
                        alert_count = {
                            "type": "count",
                            "no_helmet": self.no_helmet_counter,
                            "no_uniform": self.no_uniform_counter,
                            "no_gloves": self.no_gloves_counter,
                            "no_googles": self.no_googles_counter,
                        }
                        print('Подсчет', alert_count)

                        # await send_json_data(websocket, alert_stop_json)
                        # await send_json_data(websocket, alert_count)

                        self.no_helmet_counter = 0
                        self.no_uniform_counter = 0
                        self.no_googles_counter = 0
                        self.no_gloves_counter = 0

                        self.no_helmet_frame_counter = 0
                        self.no_uniform_frame_counter = 0
                        self.no_googles_frame_counter = 0
                        self.no_gloves_frame_counter = 0

                        self.no_helmet_detected = False
                        self.no_uniform_detected = False
                        self.no_googles_detected = False
                        self.no_gloves_detected = False

        except AttributeError:
            pass

        return json_list, alert_begin_json, alert_stop_json, alert_count

    async def handle_client(self, websocket, path):

        try:
            while True:
                frame = await websocket.recv()
                print('получил кадр')
                self.last_activity_time = datetime.datetime.now()  # Обновляем время последней активности
                asyncio.create_task(self.process_frame(frame, websocket))
        except websockets.exceptions.ConnectionClosed:
            # # Клиент отключен, выполняем необходимые действия
            # self.no_helmet_frame_counter = 0
            # self.no_uniform_frame_counter = 0
            # self.no_googles_frame_counter = 0
            # self.no_gloves_frame_counter = 0
            #
            # # подсчет unsafe_conditions (максимальное количество, которое единовременно засекли в кадре)
            # self.no_helmet_counter = 0
            # self.no_uniform_counter = 0
            # self.no_googles_counter = 0
            # self.no_gloves_counter = 0
            #
            # # состояния
            # self.no_helmet_detected = False
            # self.no_uniform_detected = False
            # self.no_googles_detected = False
            # self.no_gloves_detected = False
            #
            # self.out = None
            # self.non_detected_counter = 0  # счетчик кадров для остановки записи
            # self.frame_len = 0

            print("Клиент отключен")
            await self.wait_and_reset_state()  # Ждем 10 секунд перед сбросом состояния

    async def wait_and_reset_state(self):
        await asyncio.sleep(30)  # Ждем 10 секунд
        current_time = datetime.datetime.now()
        if (current_time - self.last_activity_time).total_seconds() >= 30:
            self.reset_state()

    def reset_state(self):
        # Сброс всех переменных состояния
        self.no_helmet_frame_counter = 0
        self.no_uniform_frame_counter = 0
        self.no_googles_frame_counter = 0
        self.no_gloves_frame_counter = 0
        self.no_helmet_counter = 0
        self.no_uniform_counter = 0
        self.no_googles_counter = 0
        self.no_gloves_counter = 0
        self.no_helmet_detected = False
        self.no_uniform_detected = False
        self.no_googles_detected = False
        self.no_gloves_detected = False
        self.out = None
        self.non_detected_counter = 0
        self.frame_len = 0

    async def process_frame(self, frame, websocket):

        frame_data = base64.b64decode(frame)
        nparr = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.frame_len += 1
        # frame = cv2.imdecode(base64.b64decode(encoded_frame), 1)

        # Обрабатываем каждый 5-й кадр
        # if asyncio.get_event_loop().time() % 1 == 0:
        if self.frame_len >= 7:
            now = datetime.datetime.now()
            formatted_now = now.strftime("%d-%m-%y-%H-%M-%S")
            print('frame_len', self.frame_len)
            # objects_data = await detect_objects(frame, formatted_now)
            objects_data, alert_begin, alert_stop, alert_count = await self.detect_objects(frame, formatted_now)
            print('отправил данные', objects_data)
            self.frame_len = 0

            if len(objects_data) > 0:
                await self.send_json_data(websocket, objects_data)
            if alert_begin is not None:
                await self.send_json_data(websocket, alert_begin)
            if alert_stop is not None:
                await self.send_json_data(websocket, alert_stop)
            if alert_count is not None:
                await self.send_json_data(websocket, alert_count)

clients = []
async def on_connect(websocket, path):
    handler = ClientHandler()
    clients.append(handler)
    print(clients)
    await handler.handle_client(websocket, path)

    # handler = ClientHandler()
    # print(handler)
    # try:
    #     while True:
    #         frame = await websocket.recv()
    #         asyncio.create_task(handler.process_frame(frame, websocket))
    # except websockets.exceptions.ConnectionClosed:
    #     print("Клиент отключен")
    #     await handler.wait_and_reset_state()  # Ждем 10 секунд перед сбросом состояния


start_server = websockets.serve(on_connect, 'localhost', 8765)
print("Сервер запущен...")

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

# active_clients = {}  # Словарь для хранения активных клиентов
#
# async def on_connect(websocket, path):
#     client_id = str(uuid.uuid4())  # Генерируем уникальный идентификатор для клиента
#     handler = ClientHandler()
#     active_clients[client_id] = handler  # Добавляем нового клиента в словарь активных клиентов
#     try:
#         while True:
#             frame = await websocket.recv()
#             await active_clients[client_id].process_frame(frame, websocket)
#     except websockets.exceptions.ConnectionClosed:
#         print("Клиент отключен")
#         await active_clients[client_id].wait_and_reset_state()  # Ждем 10 секунд перед сбросом состояния
#         del active_clients[client_id]  # Удаляем клиента из списка активных клиентов
#
# start_server = websockets.serve(on_connect, 'localhost', 8765)
# print("Сервер запущен...")
#
# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()
