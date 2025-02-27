import os
import gradio as gr
import asyncio
from chatbot import initialize_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from utils import format_ai_message_content
from datetime import datetime

# Global variables to store initialized agent and config
agent = None
agent_config = None

async def chat_with_agent(message, history):
    global agent, agent_config
    
    # Convert history into messages format that the agent expects
    messages = []
    if history:
        print("History:", history)  # Debug print
        for msg in history:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})
    
    # Add the current message
    messages.append(HumanMessage(content=message))
    
    print("Final messages:", messages)  # Debug print
    
    runnable_config = RunnableConfig(
        recursion_limit=agent_config["configurable"]["recursion_limit"],
        configurable={
            "thread_id": agent_config["configurable"]["thread_id"],
            "checkpoint_ns": "chat_mode",
            "checkpoint_id": str(datetime.now().timestamp())
        }
    )
    
    response_messages = []
    yield response_messages
    # Process message with agent
    async for chunk in agent.astream(
        {"messages": messages},  # Pass the full message history
        runnable_config
    ):
        if "agent" in chunk:
            print("agent in chunk")
            response = chunk["agent"]["messages"][0].content
            response_messages.append(dict(
                role="assistant",
                content=format_ai_message_content(response, format_mode="markdown")
            ))
            print(response_messages)
            yield response_messages
        elif "tools" in chunk:
            print("tools in chunk")
            tool_message = str(chunk["tools"]["messages"][0].content)
            response_messages.append(dict(
                role="assistant",
                content=tool_message,
                metadata={"title": "🛠️ Tool Call"}
            ))
            print(response_messages)
            yield response_messages

def train_model(dataset_url, target_column, model_type, num_gpus):
    """Handle model training request"""
    message = f"""Please train a machine learning model with these specifications:
    - Dataset: {dataset_url}
    - Target column: {target_column}
    - Model type: {model_type}
    - Number of GPUs: {num_gpus}
    """
    return message

def create_ui():
    # Create the Gradio interface
    with gr.Blocks(title="ML Training Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 ML Training Assistant")
        
        with gr.Tab("Train Model"):
            with gr.Row():
                with gr.Column():
                    # Dataset selection
                    dataset_input = gr.Textbox(
                        label="Kaggle Dataset URL",
                        placeholder="https://www.kaggle.com/datasets/example/dataset"
                    )
                    target_column = gr.Textbox(
                        label="Target Column",
                        placeholder="column_to_predict"
                    )
                    model_type = gr.Dropdown(
                        choices=[
                            "linear_regression",
                            "ridge",
                            "lasso",
                            "elastic_net",
                            "random_forest",
                            "gradient_boosting",
                            "svr",
                            "xgboost",
                            "k_neighbors"
                        ],
                        label="Model Type"
                    )
                    num_gpus = gr.Slider(
                        minimum=0,
                        maximum=8,
                        step=1,
                        value=1,
                        label="Number of GPUs"
                    )
                    train_button = gr.Button("Train Model", variant="primary")
                
                with gr.Column():
                    # Training progress and results
                    output_box = gr.TextArea(
                        label="Training Status",
                        lines=10,
                        interactive=False
                    )
                    progress_bar = gr.Progress()
        
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True
            )
            msg = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                lines=2
            )
            with gr.Row():
                submit = gr.Button("Submit")
                clear = gr.Button("Clear")

            gr.Examples(
                examples=[
                    "What machine learning models are available?",
                    "How do I prepare my dataset for training?",
                    "What are the best parameters for random forest?",
                    "Show me the available GPU options"
                ],
                inputs=msg
            )

            # Set up chat functionality
            submit.click(
                fn=chat_with_agent,
                inputs=[msg, chatbot],
                outputs=chatbot
            )
            clear.click(lambda: None, None, chatbot, queue=False)
            msg.submit(
                fn=chat_with_agent,
                inputs=[msg, chatbot],
                outputs=chatbot
            )

        # Connect the training button to the training function
        train_button.click(
            fn=train_model,
            inputs=[dataset_input, target_column, model_type, num_gpus],
            outputs=output_box,
            api_name="train_model"
        )

    return demo

async def main():
    global agent, agent_config
    # Initialize agent before creating UI
    print("Initializing agent...")
    agent_executor, config, runnable_config = await initialize_agent()
    agent = agent_executor
    agent_config = config
    
    # Create and launch the UI
    print("Starting Gradio UI...")
    demo = create_ui()
    demo.queue()
    demo.launch(share=True)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 