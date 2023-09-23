import sqlite3
import hashlib
import pyotp
import tkinter as tk


class Challenge:
    def __init__(self, name: str, render: str = "", validator: callable = lambda: True, error: str = None):
        self.name = name
        self.render = render
        self.validator = validator
        self.value = None
        self.error = error

    def validate(self, value: str) -> bool:
        return self.validator(value)

    def __str__(self):
        return self.name

    def set_value(self, value):
        self.value = value
        return self

    def copy(self):
        return Challenge(self.name, self.render, self.validator, self.error)


def usernameCheck(username: str) -> bool:
    """Length from 3 to 64 characters"""
    return 3 <= len(username) <= 64


def passwordCheck(password: str) -> bool:
    """Length from 8 characters. Must include a number, uppercase letter, and lowercase letter"""
    if len(password) < 8:
        return False
    if not any(char.isdigit() for char in password):
        return False
    if not any(char.isupper() for char in password):
        return False
    if not any(char.islower() for char in password):
        return False
    return True


def totpCheck(totp: str) -> bool:
    """Must be a 6 digit number"""
    return len(totp) == 6 and totp.isdigit()


default_challenges = {
    "username": Challenge("username", "Username", usernameCheck),
    "password": Challenge("password", "Password", passwordCheck),
    "totp": Challenge("totp", "TOTP", totpCheck)
}


class Login:
    def __init__(self):
        self.conn = sqlite3.connect("main.db")
        self.cur = self.conn.cursor()
        # Create the users table if it doesn't exist
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS "users" (
                "username" TEXT PRIMARY KEY UNIQUE NOT NULL,
                "password" TEXT NOT NULL,
                "salt" TEXT NOT NULL,
                "privilegeLevel" INTEGER NOT NULL,
                "totpSecret" TEXT
            )
        """)
        self.conn.commit()

    def attempt_sign_in(self, challenges: list[Challenge] = []) -> bool | str:
        """
        On successful sign in, returns True
        On unsuccessful sign in, returns a Challenge
        """
        challenge_names = [str(challenge) for challenge in challenges]

        # Check if a username was provided
        if "username" not in challenge_names:
            # Create a copy of the default username challenge
            username_challenge = default_challenges["username"].copy()
            username_challenge.error = "Username not provided"
            return username_challenge
        username = [challenge for challenge in challenges if str(challenge) == "username"][0].value
        # Check if the user exists
        self.cur.execute(f"SELECT * FROM users WHERE username='{username}'")
        user = self.cur.fetchone()
        if user is None:
            # Create a copy of the default username challenge
            username_challenge = default_challenges["username"].copy()
            username_challenge.error = "User not found"
            return username_challenge

        # Check if a password was provided
        if "password" not in challenge_names:
            # Create a copy of the default password challenge
            password_challenge = default_challenges["password"].copy()
            password_challenge.error = "Password not provided"
            return password_challenge
        password = [challenge for challenge in challenges if str(challenge) == "password"][0].value
        # Fetch the salt and hash the password
        salt = user[3]
        hashed_password = hashlib.sha512((password + salt).encode()).hexdigest()
        # Check if the password is correct
        if hashed_password != user[2]:
            # Create a copy of the default password challenge
            password_challenge = default_challenges["password"].copy()
            password_challenge.error = "Incorrect password"
            return password_challenge

        # Check if TOTP is enabled
        if user[5] is not None:
            # Check if a TOTP was provided
            if "totp" not in challenge_names:
                # Create a copy of the default TOTP challenge
                totp_challenge = default_challenges["totp"].copy()
                totp_challenge.error = "TOTP not provided"
                return totp_challenge
            totp = [challenge for challenge in challenges if str(challenge) == "totp"][0].value
            # Check if the TOTP is correct
            if not pyotp.TOTP(user[5]).verify(totp):
                # Create a copy of the default TOTP challenge
                totp_challenge = default_challenges["totp"].copy()
                totp_challenge.error = "Incorrect TOTP"
                return totp_challenge
        return True, user[0]


class LoginUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.challenges = []  # A list of challenge objects
        self.text_boxes = []  # A list of text boxes
        self.next_input_index = 0
        self.uid = None
        # Resize the window to 300x300
        self.master.geometry("300x200")

        self.submit_button = tk.Button(self, text="Submit", command=self.submit)
        self.submit_button.pack()
        self.login = Login()
        self.master.bind("<Return>", lambda event: self.submit())

        self.submit()

    def submit(self):
        login = Login()
        # Get the values from the text boxes
        for i in range(len(self.text_boxes)):
            self.challenges[i].set_value(self.text_boxes[i].get())
        # Attempt to sign in
        next_challenge = login.attempt_sign_in(self.challenges)
        if type(next_challenge) == tuple:
            self.master.destroy()
            self.uid = next_challenge[1]
            return True
        elif isinstance(next_challenge, Challenge):
            if next_challenge.name not in [str(challenge) for challenge in self.challenges]:
                # Add to the list of challenges
                self.challenges.append(next_challenge)
                # Add text above the text box to tell the user what to enter
                tk.Label(self, text=next_challenge.render).pack()
                # Create a new text box
                self.text_boxes.append(tk.Entry(self, **{"show": "•" if next_challenge.name == "password" else ""}))
                self.text_boxes[-1].pack()
                # Set the focus to the new text box
                self.text_boxes[-1].focus_set()
            else:
                # Print the error
                print(next_challenge.error)

        return False

def login():
    x = LoginUI(tk.Tk())
    x.mainloop()
    return x.uid

if __name__ == "__main__":
    login = Login()
    challenges = []
    next_challenge = login.attempt_sign_in(challenges)
    while isinstance(next_challenge, Challenge):
        # Check if it's a challenge already in the list
        if next_challenge.name in [str(challenge) for challenge in challenges]:
            # Ask for it again and set the value
            print(next_challenge.error)
        else:
            # This is a new challenge, add it to the list
            challenges.append(next_challenge)
        challenges[-1].set_value(input(f"{next_challenge.render} > "))
        next_challenge = login.attempt_sign_in(challenges)
    print("Signed in!")
