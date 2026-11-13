import os
import pandas as pd
from typing import Annotated, Any
from dotenv import load_dotenv
from datetime import date, timedelta
from typing import TypedDict, Literal

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# silence HF noise before the heavy imports so it can't overwrite the spinner
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

console = Console()
startup_status = console.status("[bold cyan]Loading system components, please wait...[/]", spinner="dots")
startup_status.start()

from langchain_chroma import Chroma
from langgraph.types import Command
from langchain.agents import create_agent, AgentState
from langgraph.types import interrupt
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()  # Load environment variables (e.g., GOOGLE_API_KEY) from .env

if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError("Set GOOGLE_API_KEY in your .env file")

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest"
)

# Initialize local HuggingFace embedding model for semantic product search
embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 16, "normalize_embeddings": True},
    query_encode_kwargs={
        "prompt": "Instruct: Given an inventory management query, retrieve the relevant product records, descriptions, quantities, pricing and expiry dates details that match the request\nQuery: "
    },
    show_progress=True
)

# Connect to the persisted ChromaDB collection for RAG retrieval
vector_db = Chroma(collection_name='Inventory', persist_directory="./data/vector_db/chroma_data", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

################ Helper functions for inventory management ################
def read_inventory_data(file_path:str):
    """Read inventory data from the excel file and return dataframe"""
    return pd.read_excel(file_path, parse_dates=["Product Expiry Date"])

def update_inventory_data(inventory_data:pd.DataFrame, products_from_user_query:list[dict[str, str]]):
    """Update inventory data dataframe to remove the purchased quantities from the inventory"""
    for product in products_from_user_query:
        product_name = product["product_name"]
        quantity = product["quantity"]

        # update the quantity of the product in the inventory data and return the updated dataframe
        if product_name in inventory_data["Product Name"].values:
            inventory_data.loc[inventory_data["Product Name"] == product_name, "Product Quantity"] -= int(quantity)

    return inventory_data

def check_for_out_of_stock_products(inventory_data:pd.DataFrame, quantity_threshold:int):
    """Check for products that are below the quantity threshold and return a list of out of stock products"""
    out_of_stock_products = inventory_data[inventory_data["Product Quantity"] < quantity_threshold]
    # filter the attributes to return only product name, quantity and supplier mail
    out_of_stock_products = out_of_stock_products[["Product Name", "Product Quantity", "Product Supplier Email"]]
    return out_of_stock_products.to_dict(orient="records")

def write_inventory_data(inventory_data:pd.DataFrame, file_path:str):
    """Write the updated inventory data dataframe back to the excel file"""
    inventory_data.to_excel(file_path, index=False)

def check_for_expiry_products(inventory_data:pd.DataFrame, current_date:date, expiry_check_days_threshold:int):
    """Check for products that are about to expire within specified threshold time and return a list of near expiry products"""
    expiry_cutoff_date = current_date + timedelta(days=expiry_check_days_threshold)
    near_expiry_df = inventory_data[
        (inventory_data["Product Expiry Date"].dt.date >= current_date) & 
        (inventory_data["Product Expiry Date"].dt.date <= expiry_cutoff_date)
    ]
    result_df = near_expiry_df[["Product Name", "Product Expiry Date", "Product Quantity", "Product Price"]].copy()
    result_df["Product Expiry Date"] = result_df["Product Expiry Date"].dt.strftime("%Y-%m-%d")
    return result_df.to_dict(orient="records")

############### Chat Agent Tools (RAG & Live Excel Queries) ###############
@tool
def retrieve_relevant_docs(query: str) -> str:
    """Retrieve relevant product documents from the vector database for semantic/descriptive queries."""
    docs = retriever.invoke(query)
    if docs:
        SEP = "\n\n" + "-" * 50 + "\n\n"
        return SEP.join([doc.page_content for doc in docs])
    else:
        return "No relevant documents found."

# InjectedState hides db_path from the LLM tool schema and injects it directly from InventoryAgentState at runtime
@tool
def get_product_details(product_name:str,
                        db_path: Annotated[str, InjectedState("DB_PATH")])-> str:
    """ Look up the exact live quantity, price and expiry date for a product.

    Args:
        product_name: The name or keyword of the product to search for.
    Returns:
        A string containing the products details including live quantity, price, and expiry date.
    """

    df = read_inventory_data(db_path)

    matched_products = df[df["Product Name"].str.contains(product_name.strip(), case=False, na=False, regex=False)]
    if matched_products.empty:
        return f"No products matching '{product_name}' were found in the inventory sheet."
    SEP = "\n\n" + "-" * 50 + "\n\n"
    return SEP.join([f"Product Name: {row['Product Name']}\nProduct Description: {row['Product Description']}\nQuantity: {row['Product Quantity']}\nPrice: ${row['Product Price']:.2f}\nExpiry Date: {row['Product Expiry Date'].strftime('%Y-%m-%d')}" for _, row in matched_products.iterrows()])

@tool
def get_inventory_summary(
                        report_type:Literal["stock", "price", "overview"], 
                        db_path: Annotated[str, InjectedState("DB_PATH")],
                        operator: Literal["<", "<=", ">", ">=", "="] | None = None,
                        threshold: float|None = None)-> str:
    """Filter inventory products by stock quantity or price, or get a high-level overview.

    Args:
        report_type: 'stock' to filter by quantity, 'price' to filter by price, or 'overview' for total counts.
        operator: Comparison operator ('<', '<=', '>', '>=', '='). Required for 'stock' and 'price'.
        threshold: The numerical threshold value to compare against (e.g. 5, 20.5).
    """
    df = read_inventory_data(db_path)
    if report_type == "overview":
        return f"Total product SKUs: {len(df)}, Total inventory count: {df['Product Quantity'].sum()} units."

    if operator is None or threshold is None:
        return f"Please specify both an operator ('<', '>', etc.) and a numerical threshold for a '{report_type}' report."

    # Select target column
    col = "Product Quantity" if report_type == "stock" else "Product Price"

    # Filter based on operator
    if operator == "<":
        filtered_df = df[df[col] < threshold]
    elif operator == "<=":
        filtered_df = df[df[col] <= threshold]
    elif operator == ">":
        filtered_df = df[df[col] > threshold]
    elif operator == ">=":
        filtered_df = df[df[col] >= threshold]
    elif operator == "=":
        filtered_df = df[df[col] == threshold]
    else:
        return "Invalid operator. Please use '<', '<=', '>', '>=', or '='."

    if filtered_df.empty:
        return f"No products found where {col} {operator} {threshold}."

    # Return key columns only to save LLM context
    display_cols = ["Product Name", "Product Quantity", "Product Price", "Product Expiry Date"]
    return filtered_df[display_cols].to_string(index=False)


# Define structure for the user query classification
IntentType = Literal["restock", "expiry", "chat"]

class ClassifyUserQuery(TypedDict):
    intent: IntentType

# Define structure for extracting products and their quantities from the user query
class ProductItem(TypedDict):
    product_name: str
    quantity: int

class ExtractProductsFromUserQuery(TypedDict):
    products_from_user_query: list[ProductItem]  # product name, quantity

router = llm.with_structured_output(ClassifyUserQuery)
product_extractor = llm.with_structured_output(ExtractProductsFromUserQuery)


##################### AGENT STATE ####################
class InventoryAgentState(AgentState):
    user_query: str
    message: str | None
    current_date: date  # Reference date used for expiry calculations

    # Configuration parameters injected per invocation
    DB_PATH: str  # Path to the live inventory Excel file
    EXPIRY_CHECK_DAYS_THRESHOLD: int  # Window (in days) to flag near-expiry items
    QUANTITY_THRESHOLD: int = 5  # Minimum stock level before triggering restock alerts
    SYSTEM_EMAIL_ID: str  # Sender email address used for outgoing notifications

    # Router classification result ("restock", "expiry", or "chat")
    user_query_classification: IntentType | None

    # Path A (Restock) state fields
    products_from_user_query: list[ProductItem]  # [{"product_name": str, "quantity": int}]
    empty_products: list[dict[str, Any]] | None  # [{"Product Name", "Product Quantity", "Product Supplier Email"}]
    supplier_drafts: list[dict[str, str]] | None  # [{"sender_email", "supplier_email", "subject", "body"}]

    # Path B (Expiry) state fields
    expiry_products: list[dict[str, Any]] | None  # [{"Product Name", "Product Expiry Date", "Product Quantity", "Product Price"}]
    discount_offer_drafts: list[dict[str, str]] | None  # [{"subject", "body"}]

#################### Initialize chat agent with tools and memory ####################
chat_agent_tools = [retrieve_relevant_docs, get_product_details, get_inventory_summary]
chat_agent = create_agent(model=llm,
                          tools=chat_agent_tools,
                          state_schema=InventoryAgentState,
                          system_prompt=(
                            """  
                                You are an inventory management assistant.
                                - Use get_product_details for exact quantity, price, or expiry of a specific product.
                                - Use get_inventory_summary for filters (stock/price thresholds) and overall totals.
                                - Use retrieve_relevant_docs only for descriptive or semantic questions.
                                The Excel-backed tools are the source of truth for live numbers.
                                If information is insufficient, say so. Politely decline non-inventory questions.
                                When listing several products, use a Markdown table.
                            """
                            )
                        )
    
#################### PATH A: INVENTORY UPDATE FROM RECEIPT AND SEND MAIL FOR RESTOCK PATH ####################
def extract_products(agent_state: InventoryAgentState):
    """Extract products and their quantities from the user query using llm"""
    extraction_result = product_extractor.invoke([
        SystemMessage(content="You are an inventory management agent. Extract the purchased products and their quantities. Each item must have 'product_name' (matching standard product names) and 'quantity' (as an integer)."),
        HumanMessage(content=agent_state["user_query"])
    ])
    message = f"Products extracted successfully. {extraction_result['products_from_user_query']}"

    return {
        "products_from_user_query": extraction_result["products_from_user_query"],
        "message": message,
        "messages": [AIMessage(content=message)]
    }

def handle_products_after_sold(agent_state: InventoryAgentState):
    """Search for the extracted products in the inventory, update the inventory, check for out of stocks and update the DB"""
    # Read data from the excel file
    inventory_data = read_inventory_data(agent_state["DB_PATH"])
    updated_inventory_data = update_inventory_data(inventory_data, agent_state["products_from_user_query"])

    # Write the updated inventory data back to the excel file
    write_inventory_data(updated_inventory_data, agent_state["DB_PATH"])

    # Check for out of stock products and return the list of out of stock products
    out_of_stock_products = check_for_out_of_stock_products(updated_inventory_data, agent_state["QUANTITY_THRESHOLD"])
    if out_of_stock_products:
        message = f"Inventory updated and out of stock products checked. Some products are below the quantity threshold. Out of stock products: {out_of_stock_products}"
    else:
        message = "Inventory updated and out of stock products checked. All products are above the quantity threshold."
    
    return {
        "empty_products": out_of_stock_products,
        "message": message,
        "messages": [AIMessage(content=message)]
    }

def draft_supplier_mails(agent_state: InventoryAgentState):
    """Group lowstock products by supplier and draft emails"""
    empty_products = agent_state.get("empty_products", [])

    supplier_groups = {}
    for item in empty_products:
        try:
            supplier_email = item.get("Product Supplier Email")
            if supplier_email not in supplier_groups:
                supplier_groups[supplier_email] = []
            supplier_groups[supplier_email].append(item)
        except Exception as e:
            print(f"Error processing item {item}: {e}")

    drafts = []
    for supplier_email, items in supplier_groups.items():
        items_list_str = "\n".join([f"- {i['Product Name']} (Current stock: {i['Product Quantity']})" for i in items])
        subject = "Urgent Restock Request for Low Stock Products"
        body = (
            f"Dear Supplier,\n\n"
            f"Our inventory levels for the following products have fallen below our safety threshold:\n"
            f"{items_list_str}\n\n"
            f"Please let us know your earliest delivery availability and updated pricing.\n\n"
            f"Best regards,\nSupermarket Inventory Management"
        )
        drafts.append({
            "sender_email": agent_state["SYSTEM_EMAIL_ID"],
            "supplier_email": supplier_email,
            "subject": subject,
            "body": body
        })

    message = f"Prepared {len(drafts)} supplier restock email draft(s).\n\nDrafts:\n" + "\n".join([f"To: {d['supplier_email']}, Subject: {d['subject']}, Body: {d['body']}" for d in drafts])
    return {
        "supplier_drafts": drafts,
        "message": f"Prepared {len(drafts)} supplier restock email draft(s).",
        "messages": [AIMessage(content=message)]
    }

def review_and_send_supplier_mails(agent_state: InventoryAgentState):
    """Review and send supplier emails"""

    drafts = agent_state.get("supplier_drafts", [])

    # Pause graph execution and surface drafts to the CLI; resumes when Command(resume=...) is invoked
    user_approval = interrupt({
        "question": "Do you approve sending the following supplier restock mails? (yes/no)",
        "drafts": drafts
    })

    #if user_approval.lower() in ("yes", "y", "approve", "approved", "ok"):
    if user_approval.lower() == "yes":
        for draft in drafts:
            console.print(f"[bold green]✔ Sent[/] restock email to [cyan]{draft['supplier_email']}[/]")

        message = f"Approved: Sent {len(drafts)} supplier restock emails."
        return {"message": message, "messages": [AIMessage(content=message)]}
    else:
        message = "Refused: Restock emails were not sent by manager request."
        return {"message": message, "messages": [AIMessage(content=message)]}

################### PATH B: EXPIRY CHECK AND NOTIFICATION PATH ####################
def search_for_expiry_node(agent_state: InventoryAgentState):
    """Handle expiry intent"""
    inventory_data = read_inventory_data(agent_state["DB_PATH"])
    expiry_products = check_for_expiry_products(inventory_data, agent_state["current_date"], agent_state["EXPIRY_CHECK_DAYS_THRESHOLD"])
    if expiry_products:
        message = f"Found {len(expiry_products)} products nearing expiry. products expiring: {expiry_products}"
    else:
        message = "No products nearing expiry within the specified threshold."
    
    return {
        "expiry_products": expiry_products,
        "message": message,
        "messages": [AIMessage(content=message)]
    }

def draft_discount_offer(agent_state: InventoryAgentState):
    """Draft discount offer for near expiry products"""
    expiry_products = agent_state.get("expiry_products")
    drafts = []

    for product in expiry_products:
        product_name = product["Product Name"]
        expiry_date = product["Product Expiry Date"]
        quantity = product["Product Quantity"]
        price = product["Product Price"]
        # Apply a 40% discount (customer pays 60% of the original price)
        discounted_price = price * 0.6
        subject = f"Special Discount Offer: {product_name} Nearing Expiry"
        body = (
            f"Dear Customer,\n\n"
            f"We have a special discount offer on the following product nearing expiry:\n\n"
            f"Product Name: {product_name}\n"
            f"Expiry Date: {expiry_date}\n"
            f"Quantity Available: {quantity}\n"
            f"Before Price: ${price:.2f}\n\n"
            f"After Discount Price: ${discounted_price:.2f} (40% off)\n\n"
            f"Hurry, while stocks last!\n\n"
            f"Best regards,\nSupermarket Inventory Management"
        )
        drafts.append({
            "subject": subject,
            "body": body
        })

    message = f"Prepared {len(drafts)} discount offer draft(s). Drafts:\n" + "\n".join([f"Subject: {d['subject']}, Body: {d['body']}" for d in drafts])
    return {
        "discount_offer_drafts": drafts,
        "message": message,
        "messages": [AIMessage(content=message)]
    }

def review_and_send_discount_offer(agent_state: InventoryAgentState):
    """Review and send discount offer notifications"""
    drafts = agent_state.get("discount_offer_drafts")
    # Pause graph execution and surface drafts to the CLI; resumes when Command(resume=...) is invoked
    user_approval = interrupt({
        "question": "Do you approve sending the following discount offer notifications? (yes/no)",
        "drafts": drafts
    })
    #if user_approval.lower() in ("yes", "y", "approve", "approved", "ok", "accept", "accepted"):
    if user_approval.lower() == "yes":
        for draft in drafts:
            console.print(f"[bold green]✔ Sent[/] discount offer: [magenta]{draft['subject']}[/]")

        message = f"Approved: Sent {len(drafts)} discount offer notifications."
        return {"message": message, "messages": [AIMessage(content=message)]}
    else:
        message = "Refused: Discount offer notifications were not sent by manager request."
        return {"message": message, "messages": [AIMessage(content=message)]}
#################### PATH C: CHAT PATH ####################
def chat_reply_node(agent_state: InventoryAgentState):
    return {"message": agent_state["messages"][-1].text}


def llm_route(agent_state: InventoryAgentState):
    """Classify user query to the appropriate node using llm"""
    decision = router.invoke([
        SystemMessage(content='''You are an inventory management agent.\n
        You will be given a user query and you need to classify it into one of the following intents:\n
            - restock: the user reports purchased/sold items with quantities(e.g. "sold 2 milk and 3 eggs").\n
            - expiry: the user asks you to RUN a new expiry check now (e.g. "check for near expiring products", "run expiry check").\n
            - chat: everything else, including questions ABOUT past results, history, or data (e.g. "what products were sent for expiry notification?","how much milk do we have?", "which products were bought earlier?").\n
        If unsure, choose chat.'''
       ),
        HumanMessage(content=agent_state["user_query"])
    ])
    return {
        "user_query_classification": decision["intent"],
        "messages": [HumanMessage(content=agent_state["user_query"])]
        
    }



def route_to_node(agent_state: InventoryAgentState):
    """Route user query to the appropriate node based on the classification"""

    #return name of the node to route to based on the classification
    if agent_state["user_query_classification"] is None:
        return "llm_route"
    elif agent_state["user_query_classification"] == "restock":
        return "products_extraction_node"
    elif agent_state["user_query_classification"] == "expiry":
        return "search_for_expiry_node"
    elif agent_state["user_query_classification"] == "chat":
        return "chat_node"

#################### BUILD & COMPILE LANGGRAPH WORKFLOW ####################
graph_builder = StateGraph(InventoryAgentState)

# Entry router node: classifies user intent into 'restock', 'expiry', or 'chat'
graph_builder.add_node("llm_route", llm_route)

# Path A (Restock Flow): extract sold items -> deduct from Excel -> draft & approve supplier emails
graph_builder.add_node("products_extraction_node", extract_products)
graph_builder.add_node("handle_products_after_sold_node", handle_products_after_sold)
graph_builder.add_node("draft_supplier_mails_node", draft_supplier_mails)
graph_builder.add_node("review_and_send_supplier_mails_node", review_and_send_supplier_mails)

# Path B (Expiry Flow): scan Excel for near-expiry items -> draft & approve discount offers
graph_builder.add_node("search_for_expiry_node", search_for_expiry_node)
graph_builder.add_node("draft_discount_offer_node", draft_discount_offer)
graph_builder.add_node("review_and_send_discount_offer_node", review_and_send_discount_offer)

# Path C (Chat/RAG Flow): agent with Excel & ChromaDB tools -> format final reply
graph_builder.add_node("chat_node", chat_agent)
graph_builder.add_node("chat_reply_node", chat_reply_node)

# Entry routing edges
graph_builder.add_edge(START, "llm_route")
graph_builder.add_conditional_edges(
    "llm_route",
    route_to_node,
    {
        "llm_route": "llm_route",
        "products_extraction_node": "products_extraction_node",
        "search_for_expiry_node": "search_for_expiry_node",
        "chat_node": "chat_node",
    }
)

# Path A edges: only draft supplier emails if low-stock items were detected
graph_builder.add_edge("products_extraction_node", "handle_products_after_sold_node")
graph_builder.add_conditional_edges(
    "handle_products_after_sold_node",
    lambda state: "empty_products" in state and len(state["empty_products"]) > 0,
    {
        True: "draft_supplier_mails_node",
        False: END,
    }
)
graph_builder.add_edge("draft_supplier_mails_node", "review_and_send_supplier_mails_node")
graph_builder.add_edge("review_and_send_supplier_mails_node", END)

# Path B edges: only draft discount offers if near-expiry products were found
graph_builder.add_conditional_edges(
    "search_for_expiry_node",
    lambda state: "expiry_products" in state and len(state["expiry_products"]) > 0,
    {
        True: "draft_discount_offer_node",
        False: END
    }
)
graph_builder.add_edge("draft_discount_offer_node", "review_and_send_discount_offer_node")
graph_builder.add_edge("review_and_send_discount_offer_node", END)

# Path C edges
graph_builder.add_edge("chat_node", "chat_reply_node")
graph_builder.add_edge("chat_reply_node", END)

# Use in-memory checkpointer to persist conversation history and support interrupt/resume across turns
memory = MemorySaver()
workflow = graph_builder.compile(checkpointer=memory)

# Export visual diagram of the compiled StateGraph
png_data = workflow.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

startup_status.stop()
console.print("[bold green]✔ System ready[/] [dim](graph saved to graph.png)[/]")

if __name__ == "__main__":
    # checkpointer requires a thread_id in config
    config = {"configurable": {"thread_id": "session-1"}}

    print("--- Inventory Agent Ready (type 'exit' or 'quit' to stop) ---")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Exiting...")
            break
        if not user_input:
            continue

        # Initial state to send to the graph
        initial_state = {
            "user_query": user_input,
            "DB_PATH" : "./data/inventory_data.xlsx",
            "QUANTITY_THRESHOLD" : 5,
            "EXPIRY_CHECK_DAYS_THRESHOLD" : 20,
            "SYSTEM_EMAIL_ID" : "inventory_agent@example.com",
            "current_date": date.today()
        }

        with console.status("[bold]Thinking...[/]"):
            result = workflow.invoke(initial_state, config=config)

        # Handle human-in-the-loop pause triggered by interrupt() in Path A or Path B
        if "__interrupt__" in result:
            interrupt_info = result["__interrupt__"][0].value
            drafts = interrupt_info.get("drafts", [])
            console.print(f"\n[bold yellow]⚠️  {interrupt_info['question']}[/]")

            for i, d in enumerate(drafts, 1):
                is_restock = "supplier_email" in d
                title = f"Restock Draft #{i}/{len(drafts)}" if is_restock else f"Discount Draft #{i}/{len(drafts)}"
                border = "yellow" if is_restock else "magenta"

                meta = (
                    f"[bold cyan]To:[/] {d['supplier_email']}\n"
                    f"[bold cyan]From:[/] {d.get('sender_email', 'N/A')}\n"
                    if is_restock else ""
                )
                content = (
                    f"{meta}"
                    f"[bold cyan]Subject:[/] {d.get('subject')}\n"
                    f"[dim]{'─' * 50}[/]\n"
                    f"{d.get('body')}"
                )
                console.print(Panel(content, title=title, border_style=border, expand=False))

            from rich.prompt import Prompt

            YES = ["yes", "y", "approve", "approved", "ok", "accept", "accepted"]
            NO = ["no", "n", "refuse"]

            answer = Prompt.ask("Send these notifications?", choices=YES + NO, show_choices=False)
            decision = "yes" if answer in YES else "no"

            # Resume paused graph execution at the interrupt point using the same thread_id
            result = workflow.invoke(Command(resume=decision), config=config)

        # Render final response panel
        if "message" in result:
            console.print(Panel(Markdown(result["message"]), title="Agent", border_style="green"))

