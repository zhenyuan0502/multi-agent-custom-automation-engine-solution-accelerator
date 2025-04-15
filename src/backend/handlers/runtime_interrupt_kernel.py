from typing import Any, Dict, List, Optional
import semantic_kernel as sk

# Import directly from the Kernel base model class for our handlers
from semantic_kernel.kernel_pydantic import KernelBaseModel

# Define message classes directly in this file since the imports are problematic
class GetHumanInputMessage(KernelBaseModel):
    """Message requesting input from a human."""
    content: str

class MessageBody(KernelBaseModel):
    """Simple message body class with content."""
    content: str

class GroupChatMessage(KernelBaseModel):
    """Message in a group chat."""
    body: Any
    source: str
    session_id: str
    target: str = ""
    
    def __str__(self):
        content = self.body.content if hasattr(self.body, 'content') else str(self.body)
        return f"GroupChatMessage(source={self.source}, content={content})"

class NeedsUserInputHandler:
    """Handler for capturing messages that need human input."""
    
    def __init__(self):
        self.question_for_human: Optional[GetHumanInputMessage] = None
        self.messages: List[Dict[str, Any]] = []

    async def on_message(self, message: Any, sender_type: str = "unknown_type", sender_key: str = "unknown_key") -> Any:
        """Process an incoming message."""
        print(
            f"NeedsUserInputHandler received message: {message} from sender: {sender_type}/{sender_key}"
        )
        
        if isinstance(message, GetHumanInputMessage):
            self.question_for_human = message
            self.messages.append({
                "agent": {"type": sender_type, "key": sender_key},
                "content": message.content,
            })
            print("Captured question for human in NeedsUserInputHandler")
        elif isinstance(message, GroupChatMessage):
            self.messages.append({
                "agent": {"type": sender_type, "key": sender_key},
                "content": message.body.content if hasattr(message.body, 'content') else str(message.body),
            })
            print(f"Captured group chat message in NeedsUserInputHandler - {message}")
        
        return message

    @property
    def needs_human_input(self) -> bool:
        """Check if human input is needed."""
        return self.question_for_human is not None

    @property
    def question_content(self) -> Optional[str]:
        """Get the content of the question for human."""
        if self.question_for_human:
            return self.question_for_human.content
        return None

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get captured messages and clear buffer."""
        messages = self.messages.copy()
        self.messages.clear()
        print("Returning and clearing captured messages in NeedsUserInputHandler")
        return messages

class AssistantResponseHandler:
    """Handler for capturing assistant responses."""
    
    def __init__(self):
        self.assistant_response: Optional[str] = None

    async def on_message(self, message: Any, sender_type: str = None) -> Any:
        """Process an incoming message from an assistant."""
        print(
            f"on_message called in AssistantResponseHandler with message from sender: {sender_type} - {message}"
        )
        
        if hasattr(message, "body") and sender_type in ["writer", "editor"]:
            self.assistant_response = message.body.content if hasattr(message.body, 'content') else str(message.body)
            print("Assistant response set in AssistantResponseHandler")
        
        return message

    @property
    def has_response(self) -> bool:
        """Check if response is available."""
        has_response = self.assistant_response is not None
        print(f"has_response called, returning: {has_response}")
        return has_response

    def get_response(self) -> Optional[str]:
        """Get captured response."""
        response = self.assistant_response
        print(f"get_response called, returning: {response}")
        return response

# Helper function to register handlers with a Semantic Kernel instance
def register_handlers(kernel: sk.Kernel, session_id: str) -> tuple:
    """Register interrupt handlers with a Semantic Kernel instance."""
    user_input_handler = NeedsUserInputHandler()
    assistant_handler = AssistantResponseHandler()
    
    # Register the handlers with the kernel for the given session
    handler_name = f"input_handler_{session_id}"
    response_name = f"response_handler_{session_id}"
    
    # Store handlers in kernel's memory for later retrieval
    # This is a simplified approach - in a real implementation you would use proper SK plugins
    if hasattr(kernel, "register_memory_record"):
        kernel.register_memory_record(handler_name, user_input_handler)
        kernel.register_memory_record(response_name, assistant_handler)
    else:
        # Fallback if kernel doesn't have the method
        setattr(kernel, handler_name, user_input_handler)
        setattr(kernel, response_name, assistant_handler)
    
    print(f"Registered handlers for session {session_id} with kernel")
    return user_input_handler, assistant_handler