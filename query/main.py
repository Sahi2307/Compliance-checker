from dotenv import load_dotenv
from src.controllers.chat_controller import ChatController

def main():
    load_dotenv()
    ChatController.run()

if __name__ == '__main__':
    main()