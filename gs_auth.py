import argparse
import json
import gspread

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--google_oauth', type=str, default=None, required=True, help='Path to Google OAuth Credentials json file')
    args.add_argument('--google_oauth_user', type=str, default=None, required=True, help='Path to Google OAuth Authorized User json file')
    args = args.parse_args()

    google_oauth_user = None
    gc, authorized_user = gspread.oauth_from_dict(json.load(open(args.google_oauth)), google_oauth_user)
    with open(args.google_oauth_user,'w') as fp:
        fp.write(authorized_user)
