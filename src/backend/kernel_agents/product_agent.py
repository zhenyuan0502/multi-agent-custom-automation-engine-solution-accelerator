from typing import List

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from multi_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

formatting_instructions = "Instructions: returning the output of this function call verbatim to the user in markdown. Then write AGENT SUMMARY: and then include a summary of what you did."

# Define Product tools (functions)
async def get_product_info(product_id: str) -> str:
    """Get detailed information about a product."""
    # In a real system, this would query a database or API
    info = {
        "P001": {
            "name": "Super Widget",
            "description": "A versatile widget for all your needs",
            "price": 49.99,
            "stock": 150
        },
        "P002": {
            "name": "Deluxe Gadget",
            "description": "The ultimate gadget for professionals",
            "price": 199.99,
            "stock": 75
        },
        "P003": {
            "name": "Basic Tool",
            "description": "Simple and reliable tool for everyday use",
            "price": 19.99,
            "stock": 300
        }
    }
    
    product = info.get(product_id.upper(), {"name": "Unknown Product", "description": "Product not found", "price": 0, "stock": 0})
    
    return (
        f"##### Product Information\n"
        f"**Product ID:** {product_id}\n"
        f"**Name:** {product['name']}\n"
        f"**Description:** {product['description']}\n"
        f"**Price:** ${product['price']:.2f}\n"
        f"**Stock:** {product['stock']} units\n\n"
        f"{formatting_instructions}"
    )

async def update_product_price(product_id: str, new_price: float) -> str:
    """Update the price of a product."""
    # In a real system, this would update a database
    return (
        f"##### Product Price Updated\n"
        f"**Product ID:** {product_id}\n"
        f"**New Price:** ${new_price:.2f}\n\n"
        f"The product price has been successfully updated.\n"
        f"{formatting_instructions}"
    )

async def check_product_availability(product_id: str) -> str:
    """Check the availability of a product."""
    # Mock data - in a real system this would check inventory
    availability = {
        "P001": 150,
        "P002": 75,
        "P003": 300,
        "P004": 0
    }
    
    stock = availability.get(product_id.upper(), 0)
    status = "In Stock" if stock > 0 else "Out of Stock"
    
    return (
        f"##### Product Availability\n"
        f"**Product ID:** {product_id}\n"
        f"**Status:** {status}\n"
        f"**Available Units:** {stock}\n\n"
        f"{formatting_instructions}"
    )

async def add_product_to_catalog(
    product_name: str, description: str, price: float, category: str
) -> str:
    """Add a new product to the catalog."""
    # In a real system, this would add to a database
    return (
        f"##### Product Added to Catalog\n"
        f"**Name:** {product_name}\n"
        f"**Description:** {description}\n"
        f"**Price:** ${price:.2f}\n"
        f"**Category:** {category}\n\n"
        f"The product has been successfully added to the catalog.\n"
        f"{formatting_instructions}"
    )

async def update_product_description(product_id: str, new_description: str) -> str:
    """Update the description of a product."""
    # In a real system, this would update a database
    return (
        f"##### Product Description Updated\n"
        f"**Product ID:** {product_id}\n"
        f"**New Description:** {new_description}\n\n"
        f"The product description has been successfully updated.\n"
        f"{formatting_instructions}"
    )

async def get_product_reviews(product_id: str) -> str:
    """Get reviews for a product."""
    # Mock data - in a real system this would query a database
    reviews = {
        "P001": [
            {"rating": 4, "comment": "Great product, very useful!"},
            {"rating": 5, "comment": "Exceeded my expectations."}
        ],
        "P002": [
            {"rating": 5, "comment": "Perfect for my professional needs."},
            {"rating": 4, "comment": "High quality but a bit expensive."}
        ],
        "P003": [
            {"rating": 3, "comment": "Does the job but nothing special."},
            {"rating": 4, "comment": "Good value for money."}
        ]
    }
    
    product_reviews = reviews.get(product_id.upper(), [])
    
    if not product_reviews:
        return (
            f"##### Product Reviews\n"
            f"**Product ID:** {product_id}\n\n"
            f"No reviews found for this product.\n"
            f"{formatting_instructions}"
        )
    
    review_text = "\n".join([f"- Rating: {r['rating']}/5 - \"{r['comment']}\"" for r in product_reviews])
    avg_rating = sum(r['rating'] for r in product_reviews) / len(product_reviews)
    
    return (
        f"##### Product Reviews\n"
        f"**Product ID:** {product_id}\n"
        f"**Average Rating:** {avg_rating:.1f}/5\n"
        f"**Number of Reviews:** {len(product_reviews)}\n\n"
        f"**Reviews:**\n{review_text}\n\n"
        f"{formatting_instructions}"
    )

async def compare_products(product_id1: str, product_id2: str) -> str:
    """Compare two products."""
    # Mock data - in a real system this would query a database
    products = {
        "P001": {
            "name": "Super Widget",
            "price": 49.99,
            "features": "Lightweight, Durable, Water-resistant"
        },
        "P002": {
            "name": "Deluxe Gadget",
            "price": 199.99,
            "features": "High-performance, Premium materials, Extended warranty"
        },
        "P003": {
            "name": "Basic Tool",
            "price": 19.99,
            "features": "Simple design, Easy to use, Affordable"
        }
    }
    
    product1 = products.get(product_id1.upper(), {"name": "Unknown Product", "price": 0, "features": "N/A"})
    product2 = products.get(product_id2.upper(), {"name": "Unknown Product", "price": 0, "features": "N/A"})
    
    return (
        f"##### Product Comparison\n"
        f"| Feature | {product1['name']} | {product2['name']} |\n"
        f"|---------|-----------------|------------------|\n"
        f"| Price | ${product1['price']:.2f} | ${product2['price']:.2f} |\n"
        f"| Features | {product1['features']} | {product2['features']} |\n\n"
        f"{formatting_instructions}"
    )

async def get_related_products(product_id: str) -> str:
    """Get related products for a product."""
    # Mock data - in a real system this would use a recommendation engine
    related = {
        "P001": ["P002", "P003"],
        "P002": ["P001", "P004"],
        "P003": ["P001", "P005"]
    }
    
    products = {
        "P001": "Super Widget",
        "P002": "Deluxe Gadget",
        "P003": "Basic Tool",
        "P004": "Premium Accessory",
        "P005": "Value Pack"
    }
    
    related_ids = related.get(product_id.upper(), [])
    
    if not related_ids:
        return (
            f"##### Related Products\n"
            f"**Product ID:** {product_id}\n\n"
            f"No related products found.\n"
            f"{formatting_instructions}"
        )
    
    related_products = "\n".join([f"- {pid}: {products.get(pid, 'Unknown Product')}" for pid in related_ids])
    
    return (
        f"##### Related Products\n"
        f"**Product ID:** {product_id}\n\n"
        f"**Related Products:**\n{related_products}\n\n"
        f"{formatting_instructions}"
    )

async def update_product_inventory(product_id: str, quantity: int) -> str:
    """Update the inventory for a product."""
    # In a real system, this would update a database
    return (
        f"##### Inventory Updated\n"
        f"**Product ID:** {product_id}\n"
        f"**New Quantity:** {quantity}\n\n"
        f"The product inventory has been successfully updated.\n"
        f"{formatting_instructions}"
    )

async def create_product_bundle(
    bundle_name: str, product_ids: str, bundle_price: float
) -> str:
    """Create a product bundle."""
    # In a real system, this would update a database
    return (
        f"##### Product Bundle Created\n"
        f"**Bundle Name:** {bundle_name}\n"
        f"**Products:** {product_ids}\n"
        f"**Bundle Price:** ${bundle_price:.2f}\n\n"
        f"The product bundle has been successfully created.\n"
        f"{formatting_instructions}"
    )

# Create the ProductTools function
def get_product_tools(kernel: sk.Kernel) -> List[KernelFunction]:
    """Get the list of product tools for the Product Agent."""
    product_functions = [
        (get_product_info, "Get detailed information about a product."),
        (update_product_price, "Update the price of a product."),
        (check_product_availability, "Check the availability of a product."),
        (add_product_to_catalog, "Add a new product to the catalog."),
        (update_product_description, "Update the description of a product."),
        (get_product_reviews, "Get reviews for a product."),
        (compare_products, "Compare two products."),
        (get_related_products, "Get related products for a product."),
        (update_product_inventory, "Update the inventory for a product."),
        (create_product_bundle, "Create a product bundle.")
    ]
    
    # Convert the functions to kernel functions
    kernel_functions = []
    for func, description in product_functions:
        kernel_function = kernel.register_native_function(
            function=func,
            name=func.__name__,
            description=description
        )
        kernel_functions.append(kernel_function)
    
    return kernel_functions

class ProductAgent(BaseAgent):
    """Product agent implementation using Semantic Kernel."""

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        product_tools: List[KernelFunction],
    ) -> None:
        """Initialize the Product Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            product_tools: List of tools available to this agent
        """
        super().__init__(
            agent_name="ProductAgent",
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=product_tools,
            system_message="You are a Product agent. You have knowledge about products, their specifications, pricing, availability, and features. You can provide detailed information about products, compare them, and manage product data."
        )