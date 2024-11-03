import pygame
import random

# For GUI mode Switch
GUI_FLAG = True         # True : game runs in GUI mode 
# GUI_FLAG = False        # False: game runs in CLI mode


if GUI_FLAG:
    # game window
    pygame.init()

    WIDTH, HEIGHT = 800, 600
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    FONT = pygame.font.Font(None, 36)
    
    # setup game window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ad Bidding Simulation")    
    


# initialize the bidding keywords
KEYWORDS = [chr(i) for i in range(65, 91)]  # ['A', 'B', ..., 'Z']
mean_prices = {kw: random.randint(50, 150) for kw in KEYWORDS}  # Mean price for each keyword
    

def generate_current_cost(keyword):
    '''
    Generate a current cost using a normal distribution around the mean price.
    '''
    mean_price = mean_prices[keyword]
    std_dev = mean_price * 0.2  # 20% of the mean price as standard deviation
    current_cost = max(1, random.normalvariate(mean_price, std_dev))  # Avoid negative prices
    return round(current_cost, 2)

    
def cli_mode():
    '''
    Command Line Interface mode.
    '''
    print("Welcome to the Ad Bidding CLI Simulation!")
    while True:
        # Display keywords and ask the player to pick one
        print("\nAvailable Keywords:", ", ".join(KEYWORDS))
        keyword = input("Pick a keyword to bid on (or type 'exit' to quit): ").strip().upper()
        if keyword == 'EXIT' or keyword == 'QUIT':
            break
        if keyword not in KEYWORDS:
            print("Invalid keyword! Try again.")
            continue

        # Generate current cost for the chosen keyword
        current_cost = generate_current_cost(keyword)
        print(f"Current cost for keyword '{keyword}' is generated.")

        # Ask player for a bid
        try:
            bid = float(input(f"Enter your bid for keyword '{keyword}': "))
            if bid > current_cost:
                print("You won the bid!")
            else:
                print("You lost the bid.")
            print(f"Current cost was: {current_cost}")
        except ValueError:
            print("Please enter a valid bid amount.")

def gui_mode():
    '''
    Graphical Interface mode using Pygame.
    '''
    running = True
    selected_keyword = None
    player_bid = ''
    message = ''
    keyword_rects = []
    
    # Generate positions for keyword buttons
    for i, keyword in enumerate(KEYWORDS):
        x = 50 + (i % 10) * 70
        y = 100 + (i // 10) * 50
        rect = pygame.Rect(x - 10, y - 10, 50, 40)  # Button around each keyword
        keyword_rects.append((keyword, rect))

    def draw_text(text, x, y, color=BLACK):
        '''Utility function to draw text on the screen.'''
        label = FONT.render(text, True, color)
        screen.blit(label, (x, y))
    
    while running:
        screen.fill(WHITE)
        draw_text("Pick a keyword and place a bid", 50, 50)
       
        # Draw keywords as buttons
        for keyword, rect in keyword_rects:
            pygame.draw.rect(screen, BLACK, rect, 2)  # Draw box around keyword
            draw_text(keyword, rect.x + 10, rect.y + 5)
        
        # Display selected keyword and input for bid
        if selected_keyword:
            draw_text(f"Selected Keyword: {selected_keyword}", 50, 250)
            draw_text("Enter your bid:", 50, 300)
            draw_text("> ", 50, 350)
            bid_text = FONT.render(player_bid, True, BLACK)
            screen.blit(bid_text, (70, 350))
        
        # Display message
        draw_text(message, 50, 400)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                for keyword, rect in keyword_rects:
                    if rect.collidepoint(mouse_x, mouse_y):
                        selected_keyword = keyword
                        current_cost = generate_current_cost(selected_keyword)
                        message = f"Current cost for '{selected_keyword}' is generated."
                        player_bid = ''  # Reset bid input when a new keyword is selected
                        
            elif event.type == pygame.KEYDOWN:
                if selected_keyword:
                    if event.key == pygame.K_RETURN:
                        try:
                            bid = float(player_bid)
                            if bid > current_cost:
                                message = "You won the bid!"
                            else:
                                message = "You lost the bid."
                            message += f" (Current cost: {current_cost})"
                        except ValueError:
                            message = "Please enter a valid bid amount."
                        
                        player_bid = ''  # Reset bid input
                        
                    elif event.key == pygame.K_BACKSPACE:
                        player_bid = player_bid[:-1]
                        
                    else:
                        player_bid += event.unicode

        pygame.display.flip()
    pygame.quit()
    

if __name__ == '__main__':
    
    if GUI_FLAG:
        print(f"#__Runnning the Environment in GUI mode__#")
        gui_mode()
    else :
        print(f"#__Runnning the Environment in CLI mode__#")
        cli_mode()

