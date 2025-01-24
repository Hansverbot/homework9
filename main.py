#imports
from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import torch
from transformers import pipeline


BOT_USERNAME: Final ='@hamsterhbot'

#Commands
async def start_command(update: Update, context : ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! How can i help you?")


async def help_command(update: Update, context : ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Iam a bot. Please type something so I can respond.")

async def custom(update: Update, context : ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("This is a custom command.")

async def image(update: Update, context : ContextTypes.DEFAULT_TYPE):
    await update.message.reply_photo(photo=open('hamster.jpeg', 'rb'), caption="This is me on my last vacation.")



#RESPONSES
def handle_response(text:str):
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds.",
        },
        {"role": "user", "content":text},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)[0]["generated_text"] 
    cleaned_output = outputs.replace("<|system|>\nYou are a friendly chatbot who always responds.</s>\n<|user|>\n" + text + "</s>\n<|assistant|>\n", "")

    return cleaned_output



async def handle_message(update : Update, context :ContextTypes.DEFAULT_TYPE):
    message_type=update.message.chat.type
    text :str =update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')
      
    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME,'').strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        response: str =handle_response(text)
        print('Bot:', response)
        await update.message.reply_text(response)

async def error(update: Update, context : ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')


if __name__ == "__main__":
    print("Starting bot...")
    print(torch.cuda.is_available())

    app = Application.builder().token(TOKEN).build() # Klammern hinzugefügt

    
    #Commands
    app.add_handler(CommandHandler('start',start_command))
    app.add_handler(CommandHandler('help',help_command))
    app.add_handler(CommandHandler('custom',custom))
    app.add_handler(CommandHandler('image',image))

    #messages
    app.add_handler(MessageHandler(filters.TEXT,handle_message))

    #Errors
    app.add_error_handler(error)

    print("Polling...")
    app.run_polling(poll_interval=3)